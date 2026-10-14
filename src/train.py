"""
Diabetes Prediction - Model Training Pipeline
=============================================
Loads the Pima Indians Diabetes dataset, cleans it (0s in clinical columns are
treated as missing), trains several classifiers with stratified cross-validation,
tunes the best one, and saves the final model + a metrics report (including plots)
for the Streamlit app to load.

Usage:
    python src/train.py
Outputs (in models/):
    best_model.joblib     - tuned sklearn Pipeline (preprocessing + classifier)
    metrics.json          - per-model CV + test metrics, feature importances
    roc_curve.png         - ROC curve of the tuned model
    confusion_matrix.png  - confusion-matrix heatmap of the tuned model
"""

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
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
# Order of features as they come out of the ColumnTransformer (imputed/scaled
# clinical columns first, then the passthrough columns).
PROCESSED_FEATURES = ZERO_AS_MISSING + [c for c in ALL_FEATURES if c not in ZERO_AS_MISSING]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model-specific hyperparameter grids (keyed by candidate name) for tuning.
PARAM_GRIDS = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
    },
    "Random Forest": {
        "classifier__n_estimators": [200, 400, 600],
        "classifier__max_depth": [None, 5, 10],
    },
    "Gradient Boosting": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__max_depth": [2, 3, 4],
    },
    "K-Nearest Neighbors": {
        "classifier__n_neighbors": [3, 5, 7, 9],
        "classifier__weights": ["uniform", "distance"],
    },
}


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


def evaluate(name, model, X_test, y_test) -> dict:
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
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def plot_roc(y_test, y_proba, path: str):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Tuned model (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Diabetes Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_confusion(cm, path: str):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Tuned Model)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def tune_model(X_train, y_train, base_clf, param_grid):
    gs = GridSearchCV(
        build_pipeline(base_clf),
        param_grid,
        cv=CV,
        scoring="roc_auc",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    return gs


def train():
    df = load_data()
    X = df[ALL_FEATURES]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # class_weight='balanced' counteracts the ~35% positive class imbalance
    # on the models that support it; PR-AUC is reported for all models because
    # it is more informative than ROC-AUC under imbalance.
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }

    results = []
    best_name, best_cv, best_clf = None, -1, None
    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        cv_roc = cross_val_score(pipe, X, y, cv=CV, scoring="roc_auc")
        cv_acc = cross_val_score(pipe, X, y, cv=CV, scoring="accuracy")
        pipe.fit(X_train, y_train)
        m = evaluate(name, pipe, X_test, y_test)
        m.pop("y_pred"), m.pop("y_proba")  # not JSON-serializable
        m["cv_roc_auc_mean"] = round(float(cv_roc.mean()), 4)
        m["cv_roc_auc_std"] = round(float(cv_roc.std()), 4)
        m["cv_accuracy_mean"] = round(float(cv_acc.mean()), 4)
        m["cv_accuracy_std"] = round(float(cv_acc.std()), 4)
        results.append(m)
        print(
            f"{name:22s} cv_roc={m['cv_roc_auc_mean']:.3f}±{m['cv_roc_auc_std']:.3f} "
            f"pr_auc={m['pr_auc']:.3f} acc={m['accuracy']:.3f}"
        )
        if m["cv_roc_auc_mean"] > best_cv:
            best_cv, best_name, best_clf = m["cv_roc_auc_mean"], name, clf

    # ---- Tune the winning model family with GridSearchCV ----
    param_grid = PARAM_GRIDS[best_name]
    print(f"\nTuning {best_name} ...")
    gs = tune_model(X_train, y_train, best_clf, param_grid)
    tuned = gs.best_estimator_
    tuned_metrics = evaluate(f"{best_name} (tuned)", tuned, X_test, y_test)
    y_pred, y_proba = tuned_metrics.pop("y_pred"), tuned_metrics.pop("y_proba")

    # Refit tuned model on the full dataset for maximum training signal.
    tuned.fit(X, y)

    # Feature importances (trees expose feature_importances_, linear models coef_).
    importances = {}
    clf_step = tuned.named_steps["classifier"]
    vals = None
    if hasattr(clf_step, "feature_importances_"):
        vals = clf_step.feature_importances_
    elif hasattr(clf_step, "coef_"):
        vals = np.abs(np.ravel(clf_step.coef_))
    if vals is not None and len(vals) == len(PROCESSED_FEATURES):
        importances = {PROCESSED_FEATURES[i]: round(float(v), 4) for i, v in enumerate(vals)}

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(tuned, os.path.join(MODEL_DIR, "best_model.joblib"))
    plot_roc(y_test, y_proba, os.path.join(MODEL_DIR, "roc_curve.png"))
    plot_confusion(confusion_matrix(y_test, y_pred), os.path.join(MODEL_DIR, "confusion_matrix.png"))

    report = {
        "best_model": f"{best_name} (tuned)",
        "best_params": gs.best_params_,
        "best_cv_roc_auc": round(float(gs.best_score_), 4),
        "models": sorted(results, key=lambda r: r["cv_roc_auc_mean"], reverse=True),
        "tuned_test_metrics": tuned_metrics,
        "feature_importances": importances,
        "feature_names": ALL_FEATURES,
        "dataset_rows": int(len(df)),
        "positive_rate": round(float(y.mean()), 4),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nBest model: {report['best_model']} (CV ROC-AUC {report['best_cv_roc_auc']:.3f})")
    print(f"Tuned test ROC-AUC={tuned_metrics['roc_auc']:.3f} PR-AUC={tuned_metrics['pr_auc']:.3f}")
    print(f"Saved model + plots to {MODEL_DIR}")
    return report


if __name__ == "__main__":
    train()
