"""
Diabetes Prediction Using Machine Learning - Streamlit App
==========================================================
Interactive web app that loads the trained Gradient Boosting model and lets a
user enter patient measurements to get a diabetes risk prediction, plus a view
of model performance on the Pima Indians Diabetes dataset.

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
        with cols[i % 2]:
            inputs[feat] = st.slider(feat, lo, hi, (lo + hi) / 2, step, help=help_text)

    if st.button("Predict", type="primary", use_container_width=True):
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
            st.dataframe(row.T.rename(columns={0: "value"}), use_container_width=True)

        with st.expander("How confident is the model?"):
            st.write(
                f"The best model (**{metrics['best_model']}**) achieved a ROC-AUC of "
                f"**{metrics['best_roc_auc']:.3f}** on a held-out test set. "
                "A probability above 50% is classified as diabetic; adjust your "
                "inputs to see how risk changes."
            )


def render_performance(metrics):
    st.subheader("Model comparison")
    st.write(
        f"Trained on **{metrics['dataset_rows']}** patient records "
        f"(~{metrics['positive_rate']:.0%} diabetic). All models share the same "
        "preprocessing: median imputation for missing clinical values, then "
        "standard scaling."
    )

    df = pd.DataFrame(metrics["models"])[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    st.dataframe(df.style.format({"accuracy": "{:.3f}", "precision": "{:.3f}",
                                  "recall": "{:.3f}", "f1": "{:.3f}", "roc_auc": "{:.3f}"}),
                 use_container_width=True)

    st.info(f"🏆 Best model by ROC-AUC: **{metrics['best_model']}** "
            f"(ROC-AUC {metrics['best_roc_auc']:.3f})")

    # Bar chart of ROC-AUC per model.
    chart_df = df.set_index("model")[["roc_auc"]]
    st.bar_chart(chart_df)

    st.subheader("Confusion matrix of the best model")
    best = metrics["models"][0]
    cm = pd.DataFrame(
        best["confusion_matrix"],
        index=["Actual: No", "Actual: Yes"],
        columns=["Predicted: No", "Predicted: Yes"],
    )
    st.dataframe(cm, use_container_width=True)


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
3. Train 4 classifiers (Logistic Regression, Random Forest, Gradient Boosting, KNN).
4. Select the best by ROC-AUC and save it as a reusable `sklearn` pipeline.

**Tech stack:** Python, scikit-learn, pandas, Streamlit.

*Disclaimer: This is a machine-learning demo for educational purposes and is **not**
a medical device. Do not use it for clinical diagnosis.*
"""
    )


if __name__ == "__main__":
    main()
