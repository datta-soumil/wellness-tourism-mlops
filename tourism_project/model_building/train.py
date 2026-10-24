"""Tune, evaluate, track, and serialize the customer-purchase model."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEPLOYMENT_DIR = PROJECT_DIR / "deployment"
TARGET = "ProdTaken"

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
except ImportError:
    mlflow = None


def load_splits():
    X_train = pd.read_csv(ARTIFACT_DIR / "X_train.csv")
    X_test = pd.read_csv(ARTIFACT_DIR / "X_test.csv")
    y_train = pd.read_csv(ARTIFACT_DIR / "y_train.csv")[TARGET]
    y_test = pd.read_csv(ARTIFACT_DIR / "y_test.csv")[TARGET]
    return X_train, X_test, y_train, y_test


def main() -> None:
    X_train, X_test, y_train, y_test = load_splits()
    categorical = X_train.select_dtypes(include="object").columns.tolist()
    numeric = X_train.select_dtypes(exclude="object").columns.tolist()
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)),
    ])
    parameter_grid = {
        "model__n_estimators": [150, 300],
        "model__max_depth": [None, 12],
        "model__min_samples_split": [2, 5],
    }
    search = GridSearchCV(
        pipeline, parameter_grid, scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), n_jobs=-1, verbose=1,
    )

    if mlflow is not None:
        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_DIR / 'mlflow.db').resolve()}")
        mlflow.set_experiment("tourism-package-prediction")
        run_context = mlflow.start_run(run_name="random-forest-grid-search")
    else:
        from contextlib import nullcontext
        run_context = nullcontext()
        print("MLflow is not installed locally; metrics will be saved to JSON. "
              "GitHub Actions installs MLflow and records the run.")

    with run_context:
        search.fit(X_train, y_train)
        predictions = search.predict(X_test)
        probabilities = search.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
            "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
            "f1": round(f1_score(y_test, predictions, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        }
        if mlflow is not None:
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_roc_auc", round(search.best_score_, 4))
            mlflow.log_metrics(metrics)
            input_example = X_train.head(5)
            integer_columns = input_example.select_dtypes(include="integer").columns
            input_example = input_example.astype(
                {column: "float64" for column in integer_columns}
            )
            signature = infer_signature(
                input_example, search.best_estimator_.predict(input_example)
            )
            mlflow.sklearn.log_model(
                search.best_estimator_, name="model",
                input_example=input_example, signature=signature,
                serialization_format="cloudpickle",
            )

    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, DEPLOYMENT_DIR / "model.joblib")
    (DEPLOYMENT_DIR / "metrics.json").write_text(json.dumps({
        "best_parameters": search.best_params_, "best_cv_roc_auc": round(search.best_score_, 4),
        "test_metrics": metrics,
    }, indent=2))
    print("Best parameters:", search.best_params_)
    print("Test metrics:", metrics)
    print("Classification report:\n", classification_report(y_test, predictions, digits=3))
    print(f"Saved deployable model to {DEPLOYMENT_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
