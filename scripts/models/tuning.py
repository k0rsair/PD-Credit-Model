from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

param_spaces = {
    "LogisticRegression": {
        "classifier__C": uniform(0.01, 10.0),
        "classifier__solver": ["lbfgs", "liblinear"],
    },
    "RandomForest": {
        "classifier__n_estimators": randint(50, 300),
        "classifier__max_depth": randint(3, 10),
        "classifier__min_samples_split": randint(2, 10),
        "classifier__min_samples_leaf": randint(1, 5),
    },
    "GradientBoosting": {
        "classifier__n_estimators": randint(50, 300),
        "classifier__max_depth": randint(2, 6),
        "classifier__learning_rate": uniform(0.01, 0.2),
        "classifier__subsample": uniform(0.7, 0.3),
    },
    "XGBoost": {
        "classifier__n_estimators": randint(100, 400),
        "classifier__max_depth": randint(3, 8),
        "classifier__learning_rate": uniform(0.01, 0.2),
        "classifier__subsample": uniform(0.7, 0.3),
        "classifier__colsample_bytree": uniform(0.6, 0.4),
    },
    "LightGBM": {
        "classifier__n_estimators": randint(100, 400),
        "classifier__num_leaves": randint(20, 50),
        "classifier__max_depth": randint(3, 8),
        "classifier__learning_rate": uniform(0.01, 0.2),
        "classifier__subsample": uniform(0.7, 0.3),
    },
}


def tune_hyperparameters(pipeline, model_name, X_train, y_train,
                         scoring="roc_auc", n_iter=10, cv=3, n_jobs=-1):
    """
    Подбор гиперпараметров с помощью RandomizedSearchCV.
    Возвращает обученный best_estimator_ и словарь лучших параметров.
    """
    param_distributions = param_spaces[model_name] if model_name in param_spaces else {}

    if param_distributions:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            verbose=1,
            n_jobs=n_jobs,
            random_state=42,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_
    else:
        pipeline.fit(X_train, y_train)
        return pipeline, {}
