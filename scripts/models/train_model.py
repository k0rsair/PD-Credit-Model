import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import sys

from preprocessing import get_preprocess_data
from tuning import tune_hyperparameters
from utils import evaluate_model, log_model_to_mlflow
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test, preprocessor):
    """Обучает модель с RandomizedSearchCV (tuning), считает метрики (utils.evaluate_model),
    и логирует всё в один MLflow run, включая ROC-кривую."""
    # Настраиваем пайплайн
    pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                               ("classifier", model)])

    # Пространства гиперпараметров по имени модели

    # Подбор гиперпараметров (если задано пространство); иначе — обычное обучение
    best_clf, best_params = tune_hyperparameters(
        pipeline, model_name, X_train, y_train,
        scoring="roc_auc", n_iter=10, cv=3, n_jobs=-1
    )

    # Метрики (используем utils.evaluate_model)
    metrics = evaluate_model(best_clf, X_test, y_test)
    # Логируем всё в MLflow
    log_model_to_mlflow(model_name, best_clf, metrics, best_params, X_test, y_test)
    return best_clf, metrics


def main():
    # === Получаем путь к данным из аргумента ===
    if len(sys.argv) < 2:
        print(sys.argv)
        print("Usage: python train_model.py <path_to_csv>")
        sys.exit(1)

    # === Настройка MLflow ===
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("pd_model_experiments")

    # === Загрузка данных ===
    data_path = sys.argv[1]
    df = pd.read_csv(data_path)

    target = "default.payment.next.month"
    X = df.drop(columns=[target])
    y = df[target]

    # === Разделение на train/test ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # === Препроцессинг ===
    preprocessor = get_preprocess_data(X)

    # === Модели ===
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(objective="binary:logistic", eval_metric="auc", random_state=42),
        "LightGBM": LGBMClassifier(objective="binary", metric="auc", random_state=42, verbose=-1)
    }

    # === Обучение и логирование ===
    os.makedirs("models", exist_ok=True)
    results = {}
    for name, model in models.items():
        clf, metrics = train_and_log_model(name, model, X_train, X_test, y_train, y_test, preprocessor)
        results[name] = metrics
        joblib.dump(clf, f"models/{name}.pkl")

    for name, m in results.items():
        print(f"{name}: ROC-AUC={m['roc_auc']:.3f}, F1={m['f1']:.3f}")

    best_model_name = max(results, key=lambda x: results[x]["roc_auc"])
    print(f"\nЛучшая модель: {best_model_name}")


if __name__ == "__main__":
    main()
