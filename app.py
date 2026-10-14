"""
Diabetes Prediction Using Machine Learning - Streamlit App
==========================================================
Interactive web app that loads the tuned model and lets a user enter patient
measurements to get a diabetes risk prediction, plus a view of model performance
(CV metrics, ROC curve, confusion matrix, feature importance) on the Pima
Indians Diabetes dataset.

Run:
    streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
ROC_PATH = os.path.join(BASE_DIR, "models", "roc_curve.png")
CM_PATH = os.path.join(BASE_DIR, "models", "confusion_matrix.png")

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

# (label, min, max, step, help) for each feature's slider.
INPUT_SPECS = {
    "Pregnancies": (0, 17, 1, "Number of times pregnant"),
    "Glucose": (0, 200, 1, "Plasma glucose concentration (mg/dL)"),
    "BloodPressure": (0, 122, 1, "Diastolic blood pressure (mm Hg)"),
    "SkinThickness": (0, 99, 1, "Triceps skin fold thickness (mm)"),
    "Insulin": (0, 846, 1, "2-Hour serum insulin (mu U/ml)"),
    "BMI": (0.0, 67.1, 0.1, "Body mass index (weight in kg/(height in m)^2)"),
    "DiabetesPedigreeFunction": (0.078, 2.42, 0.001, "Diabetes pedigree function (family history)"),
    "Age": (21, 81, 1, "Age in years"),
}

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
        # Train on first run (e.g. on Streamlit Cloud) if artifacts are absent.
        from src.train import train

        train()
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return model, metrics


def main():
    model, metrics = load_artifacts()

    st.title("🩺 Diabetes Prediction")
    st.caption(
        "Predict the likelihood of diabetes from clinical measurements using a "
        "machine-learning model trained on the Pima Indians Diabetes dataset."
    )

    tab_predict, tab_performance, tab_about = st.tabs(
        ["🔎 Predict", "📊 Model Performance", "ℹ️ About"]
    )

    with tab_predict:
        render_predict(model, metrics)
    with tab_performance:
        render_performance(metrics)
    with tab_about:
        render_about()


def render_predict(model, metrics):
    st.subheader("Enter patient measurements")

    inputs = {}
    cols = st.columns(2)
    for i, feat in enumerate(FEATURES):
        lo, hi, step, help_text = INPUT_SPECS[feat]
        # Midpoint default must match the min/max type (int sliders reject floats).
        mid = (lo + hi) / 2
        if isinstance(lo, int) and isinstance(hi, int):
            mid = int(round(mid))
        with cols[i % 2]:
            inputs[feat] = st.slider(feat, lo, hi, mid, step, help=help_text)

    if st.button("Predict", type="primary", width="stretch"):
        row = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)
        proba = float(model.predict_proba(row)[0, 1])
        pred = int(proba >= 0.5)

        st.divider()
        left, right = st.columns(2)
        with left:
            if pred == 1:
                st.error("🔴 **Prediction: Diabetic**")
            else:
                st.success("🟢 **Prediction: Not Diabetic**")
            st.metric("Diabetes probability", f"{proba:.1%}")
            st.progress(proba)
        with right:
            st.markdown("**Input summary**")
            st.dataframe(row.T.rename(columns={0: "value"}), width="stretch")

        tuned = metrics.get("tuned_test_metrics", {})
        with st.expander("How reliable is this model?"):
            st.write(
                f"The tuned **{metrics['best_model']}** achieved a cross-validated "
                f"ROC-AUC of **{metrics['best_cv_roc_auc']:.3f}** (5-fold) and, on a "
                f"held-out test set, ROC-AUC **{tuned.get('roc_auc', 0):.3f}** / "
                f"PR-AUC **{tuned.get('pr_auc', 0):.3f}**. "
                "A probability above 50% is classified as diabetic; adjust your "
                "inputs to see how risk changes."
            )


def render_performance(metrics):
    st.subheader("Model comparison (5-fold stratified cross-validation)")
    st.write(
        f"Trained on **{metrics['dataset_rows']}** patient records "
        f"(~{metrics['positive_rate']:.0%} diabetic). All models share the same "
        "preprocessing: median imputation for missing clinical values, then standard "
        "scaling. Trees/linear models use balanced class weights to counter the "
        "imbalance; **PR-AUC** is reported because it is more informative than "
        "ROC-AUC under class imbalance."
    )

    df = pd.DataFrame(metrics["models"])
    df["cv_roc_auc"] = df.apply(
        lambda r: f"{r['cv_roc_auc_mean']:.3f} ± {r['cv_roc_auc_std']:.3f}", axis=1
    )
    show = df[
        ["model", "cv_roc_auc", "cv_accuracy_mean", "pr_auc", "roc_auc", "f1"]
    ].rename(
        columns={
            "cv_accuracy_mean": "cv_accuracy",
            "roc_auc": "test_roc_auc",
            "f1": "test_f1",
        }
    )
    st.dataframe(
        show.style.format(
            {"cv_accuracy_mean": "{:.3f}", "cv_accuracy": "{:.3f}",
             "pr_auc": "{:.3f}", "test_roc_auc": "{:.3f}", "test_f1": "{:.3f}"}
        ),
        width="stretch",
    )

    tuned = metrics.get("tuned_test_metrics", {})
    st.info(
        f"🏆 Best model: **{metrics['best_model']}** — tuned params "
        f"`{metrics.get('best_params', {})}`. "
        f"CV ROC-AUC **{metrics['best_cv_roc_auc']:.3f}**."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{tuned.get('accuracy', 0):.3f}")
    c2.metric("Test ROC-AUC", f"{tuned.get('roc_auc', 0):.3f}")
    c3.metric("Test PR-AUC", f"{tuned.get('pr_auc', 0):.3f}")
    c4.metric("Test F1", f"{tuned.get('f1', 0):.3f}")

    st.subheader("Feature importance")
    fi = metrics.get("feature_importances", {})
    if fi:
        fi_sorted = dict(sorted(fi.items(), key=lambda kv: kv[1], reverse=True))
        st.bar_chart(pd.Series(fi_sorted))
        st.caption(
            "Absolute model weight per feature (coefficients for linear models, "
            "impurity-based importance for trees)."
        )
    else:
        st.caption("Feature importances not available for this model.")

    left, right = st.columns(2)
    with left:
        st.subheader("ROC curve")
        if os.path.exists(ROC_PATH):
            st.image(ROC_PATH, width="stretch")
    with right:
        st.subheader("Confusion matrix")
        if os.path.exists(CM_PATH):
            st.image(CM_PATH, width="stretch")


def render_about():
    st.subheader("About this project")
    st.markdown(
        """
**Diabetes Prediction Using Machine Learning**

This project builds a binary classifier for the
[Pima Indians Diabetes dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
(768 patients, 8 clinical features).

**Pipeline**
1. Treat `0` as *missing* in clinical columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI).
2. Median imputation + standard scaling.
3. Train 4 classifiers (Logistic Regression, Random Forest, Gradient Boosting, KNN)
   with **5-fold stratified cross-validation**.
4. **Tune** the best model family with `GridSearchCV`.
5. Select by cross-validated ROC-AUC and save a reusable `sklearn` pipeline.

**Metrics** include accuracy, precision, recall, F1, ROC-AUC and **PR-AUC** (the
latter being more informative for the imbalanced, medical-imbalanced setting).

**Tech stack:** Python, scikit-learn, pandas, Streamlit.

*Disclaimer: This is a machine-learning demo for educational purposes and is **not**
a medical device. Do not use it for clinical diagnosis.*
"""
    )


if __name__ == "__main__":
    main()
