"""
Diabetes Prediction - Model Training Pipeline
=============================================
Loads the Pima Indians Diabetes dataset, cleans it (0s in clinical columns are
treated as missing), trains several classifiers, evaluates them, and saves the
best-performing model + a metrics report for the Streamlit app to load.

Usage:
    python src/train.py
Outputs:
    models/best_model.joblib   - sklearn Pipeline (preprocessing + classifier)
    models/metrics.json        - per-model metrics + dataset summary
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# In the Pima dataset these clinical columns use 0 to encode a missing value.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
ALL_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Replace the 0-encodes-missing values with NaN so the imputer can fill them.
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    return df


def build_pipeline(model) -> Pipeline:
    """Preprocess (impute missing -> scale) then classify, as one pipeline so the
    saved artifact can predict directly from raw user input."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "impute_scale",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(ZERO_AS_MISSING),
            ),
            (
                "passthrough",
                "passthrough",
                [c for c in ALL_FEATURES if c not in ZERO_AS_MISSING],
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", model)])


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train():
    df = load_data()
    X = df[ALL_FEATURES]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }

    results = []
    best_name, best_score, best_pipe = None, -1, None
    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)
        metrics = evaluate(name, pipe, X_test, y_test)
        results.append(metrics)
        if metrics["roc_auc"] > best_score:
            best_score, best_name, best_pipe = metrics["roc_auc"], name, pipe
        print(
            f"{name:22s} acc={metrics['accuracy']:.3f} "
            f"f1={metrics['f1']:.3f} roc_auc={metrics['roc_auc']:.3f}"
        )

    # Refit best model on the full dataset for maximum training signal.
    best_pipe.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_pipe, os.path.join(MODEL_DIR, "best_model.joblib"))

    report = {
        "best_model": best_name,
        "best_roc_auc": best_score,
        "models": sorted(results, key=lambda r: r["roc_auc"], reverse=True),
        "feature_names": ALL_FEATURES,
        "dataset_rows": int(len(df)),
        "positive_rate": round(float(y.mean()), 4),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nBest model: {best_name} (ROC-AUC {best_score:.3f})")
    print(f"Saved to {os.path.join(MODEL_DIR, 'best_model.joblib')}")
    return report


if __name__ == "__main__":
    train()
