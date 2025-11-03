import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score,roc_curve
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    """Возвращает словарь метрик."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

def log_model_to_mlflow(model_name, model, metrics, best_params, X_test, y_test):
    # ROC-кривая для артефактов
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
    except Exception:
        # на случай отсутствия predict_proba
        fpr = tpr = None

    """Логирует результаты и модель в MLflow."""
    with mlflow.start_run(run_name=model_name):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        if best_params:
            mlflow.log_params(best_params)
        # метрики
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # ROC plot
        if fpr is not None and tpr is not None:
            plt.figure()
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={metrics['roc_auc']:.3f})")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model_name}")
            plt.legend()
            os.makedirs("artifacts", exist_ok=True)
            plot_path = f"artifacts/roc_{model_name}.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)

        mlflow.sklearn.log_model(model, name=f"{model_name}_model")

        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name.lower()}_best.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"{model_name}: ROC-AUC={metrics['roc_auc']:.4f}")