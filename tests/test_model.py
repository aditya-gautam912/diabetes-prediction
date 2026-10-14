"""Unit tests for the trained diabetes model and its metrics report."""

import json
import os

import joblib
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# A patient matching the dataset's first (diabetic) row.
HIGH_RISK = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
# A low-risk patient.
LOW_RISK = [1, 85, 66, 29, 0, 26.6, 0.351, 31]


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


def test_artifacts_exist():
    assert os.path.exists(MODEL_PATH), "best_model.joblib missing - run src/train.py"
    assert os.path.exists(METRICS_PATH), "metrics.json missing - run src/train.py"


def test_metrics_structure():
    with open(METRICS_PATH) as f:
        m = json.load(f)
    for key in ["best_model", "best_cv_roc_auc", "models", "tuned_test_metrics",
                "feature_importances", "dataset_rows"]:
        assert key in m, f"metrics.json missing key: {key}"
    assert m["dataset_rows"] == 768
    # Every model should report cross-validated and imbalance-aware metrics.
    for row in m["models"]:
        assert "cv_roc_auc_mean" in row and "pr_auc" in row


def test_prediction_probability_in_range(model):
    for patient in (HIGH_RISK, LOW_RISK):
        row = pd.DataFrame([patient], columns=FEATURES)
        proba = float(model.predict_proba(row)[0, 1])
        assert 0.0 <= proba <= 1.0


def test_high_riskoutranks_low_risk(model):
    high = float(model.predict_proba(pd.DataFrame([HIGH_RISK], columns=FEATURES))[0, 1])
    low = float(model.predict_proba(pd.DataFrame([LOW_RISK], columns=FEATURES))[0, 1])
    assert high > low


def test_prediction_is_binary(model):
    pred = int(model.predict(pd.DataFrame([HIGH_RISK], columns=FEATURES))[0])
    assert pred in (0, 1)
