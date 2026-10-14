"""
Diabetes Prediction Using Machine Learning - Streamlit App
==========================================================
Interactive web app that loads the tuned model and lets a user enter patient
measurements to get a diabetes risk prediction (with a color-coded risk gauge),
plus a view of model performance (CV metrics, ROC curve, confusion matrix,
feature importance) on the Pima Indians Diabetes dataset.

Run:
    streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import plotly.express as px
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

# Logical groupings for the input UI.
GROUPS = {
    "Vitals": ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"],
    "Reproductive & Family History": ["Pregnancies", "DiabetesPedigreeFunction", "Age"],
}

# (min, max, step, help) for each feature's slider.
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

PRESETS = {
    "Load high-risk example": {
        "Pregnancies": 8, "Glucose": 180, "BloodPressure": 74, "SkinThickness": 35,
        "Insulin": 200, "BMI": 42.0, "DiabetesPedigreeFunction": 1.0, "Age": 60,
    },
    "Load low-risk example": {
        "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29,
        "Insulin": 80, "BMI": 26.6, "DiabetesPedigreeFunction": 0.35, "Age": 31,
    },
}

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

# Color palettes for Day (light) and Night (dark) modes.
THEMES = {
    "light": {
        "bg": "#ffffff", "bg2": "#f0f2f6", "text": "#262730", "muted": "#6c757d",
        "card": "#ffffff", "border": "#e6e9ef", "accent": "#0d6efd", "marker": "#111111",
    },
    "dark": {
        "bg": "#0e1117", "bg2": "#161b22", "text": "#e6eaf0", "muted": "#9aa4b2",
        "card": "#161b22", "border": "#2a3038", "accent": "#4c9aff", "marker": "#ffffff",
    },
}


def build_css(t: dict) -> str:
    return f"""
    <style>
      /* ---- App surfaces ---- */
      .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
      [data-testid="stToolbar"] {{ background-color: {t['bg']}; color: {t['text']}; }}
      :root, .stApp {{
        --background-color: {t['bg']};
        --secondary-background-color: {t['bg2']};
        --text-color: {t['text']};
        --primary-color: {t['accent']};
      }}
      .main .block-container {{ background-color: {t['bg']}; color: {t['text']}; }}
      [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{ background-color: {t['bg2']}; }}
      [data-testid="stSidebar"] * {{ color: {t['text']}; }}

      /* Text */
      h1, h2, h3, h4, p, li, label, .stMarkdown, [data-testid="stMarkdown"] p {{ color: {t['text']}; }}
      hr {{ border-color: {t['border']}; }}

      /* Custom header + sections */
      .app-header {{ font-size: 2.1rem; font-weight: 800; margin-bottom: 0; letter-spacing: -0.5px; color: {t['text']}; }}
      .app-header .accent {{ color: {t['accent']}; }}
      .app-tagline {{ color: {t['muted']}; font-size: 0.95rem; margin-top: 0.1rem; margin-bottom: 0.4rem; }}
      .group-title {{ font-size: 1.05rem; font-weight: 700; color: {t['accent']};
                      border-left: 4px solid {t['accent']}; padding-left: 8px; margin: 4px 0 2px; }}
      .riskbar {{ position: relative; height: 16px; border-radius: 8px; margin: 10px 0 4px;
                  background: linear-gradient(to right, #2e7d32 0%, #2e7d32 30%, #f9a825 30%, #f9a825 70%, #c62828 70%, #c62828 100%); }}
      .riskbar .marker {{ position: absolute; top: -5px; width: 5px; height: 26px; background: {t['marker']};
                          border-radius: 3px; transform: translateX(-50%); box-shadow: 0 0 5px rgba(0,0,0,.55); }}
      .risklabels {{ display: flex; justify-content: space-between; color: {t['muted']}; font-size: 0.8rem; margin-bottom: 8px; }}
      .big-pct {{ font-size: 2.6rem; font-weight: 800; line-height: 1; }}

      /* Widgets */
      [data-testid="stButton"] > button {{ background-color: {t['card']}; color: {t['text']}; border: 1px solid {t['border']}; }}
      [data-testid="stButton"] > button:hover {{ border-color: {t['accent']}; color: {t['accent']}; }}
      [data-testid="stButton"] > button[kind="primary"] {{ background-color: {t['accent']}; color: #ffffff; border: none; }}
      [data-testid="stMetric"] {{ background-color: {t['card']}; border: 1px solid {t['border']}; padding: 10px; border-radius: 10px; }}
      [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {{ color: {t['text']}; }}
      [data-testid="stExpander"] details {{ background-color: {t['card']}; border: 1px solid {t['border']}; border-radius: 10px; }}
      [data-testid="stExpander"] summary {{ color: {t['text']}; }}
      [data-testid="stAlert"] {{ background-color: {t['card']}; }}
      [data-testid="stRadio"] label, [data-testid="stRadio"] * {{ color: {t['text']}; }}
      [data-testid="stDataFrame"] iframe {{ background-color: {t['card']}; }}
    </style>
    """


def inject_css(theme: str):
    st.markdown(build_css(THEMES[theme]), unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
        from src.train import train

        train()
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return model, metrics


def risk_zone(p: float):
    if p < 0.30:
        return "Low", "#2e7d32", st.success
    if p < 0.70:
        return "Moderate", "#f9a825", st.warning
    return "High", "#c62828", st.error


def main():
    # Day / Night toggle (sidebar)
    with st.sidebar:
        st.markdown("### Appearance")
        choice = st.radio(
            "Theme", ["☀️ Day", "🌙 Night"], horizontal=True,
            label_visibility="collapsed", key="theme_choice",
        )
    theme = "dark" if choice == "🌙 Night" else "light"

    inject_css(theme)
    model, metrics = load_artifacts()

    st.markdown('<div class="app-header">🩺 Diabetes <span class="accent">Risk Predictor</span></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="app-tagline">Machine-learning risk assessment · Pima Indians dataset · '
        'educational demo, <b>not a medical device</b></div>',
        unsafe_allow_html=True,
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


def apply_preset(values: dict):
    for feat, v in values.items():
        st.session_state[feat] = v


def render_predict(model, metrics):
    # Preset buttons (callbacks run before the sliders render).
    p1, p2, _ = st.columns([1, 1, 3])
    with p1:
        st.button("🔴 High-risk example", on_click=apply_preset, args=(PRESETS["Load high-risk example"],),
                  width="stretch")
    with p2:
        st.button("🟢 Low-risk example", on_click=apply_preset, args=(PRESETS["Load low-risk example"],),
                  width="stretch")

    # Seed slider defaults into session state so presets can override them cleanly
    # (passing both `value=` and a session-state key triggers a Streamlit warning).
    for feat in FEATURES:
        if feat not in st.session_state:
            lo, hi, _, _ = INPUT_SPECS[feat]
            mid = (lo + hi) / 2
            if isinstance(lo, int) and isinstance(hi, int):
                mid = int(round(mid))
            st.session_state[feat] = mid

    inputs = {}
    for group, feats in GROUPS.items():
        st.markdown(f'<div class="group-title">{group}</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, feat in enumerate(feats):
            lo, hi, step, help_text = INPUT_SPECS[feat]
            with cols[i % 2]:
                inputs[feat] = st.slider(feat, lo, hi, step=step, help=help_text, key=feat)

    st.divider()
    if st.button(" Predict", type="primary", width="stretch"):
        row = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)
        proba = float(model.predict_proba(row)[0, 1])
        zone, color, banner = risk_zone(proba)

        # Risk gauge
        st.markdown(f'<div class="big-pct" style="color:{color}">{proba:.0%} '
                    f'<span style="font-size:1.1rem;color:#6c757d">· {zone} risk</span></div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="riskbar"><div class="marker" style="left:{proba * 100:.1f}%"></div></div>'
            f'<div class="risklabels"><span>Low (&lt;30%)</span><span>Moderate (30–70%)</span>'
            f'<span>High (&gt;70%)</span></div>',
            unsafe_allow_html=True,
        )
        if zone == "Low":
            banner(f"🟢 **{zone} risk** — model reads this patient as unlikely to be diabetic.")
        elif zone == "Moderate":
            banner(f"🟠 **{zone} risk** — some indicators are elevated; worth monitoring.")
        else:
            banner(f"🔴 **{zone} risk** — multiple indicators point toward diabetes.")

        with st.expander("Input summary"):
            st.dataframe(row.T.rename(columns={0: "value"}), width="stretch")

        tuned = metrics.get("tuned_test_metrics", {})
        with st.expander("How reliable is this model?"):
            st.write(
                f"The tuned **{metrics['best_model']}** achieved a cross-validated ROC-AUC of "
                f"**{metrics['best_cv_roc_auc']:.3f}** (5-fold) and, on a held-out test set, "
                f"ROC-AUC **{tuned.get('roc_auc', 0):.3f}** / PR-AUC **{tuned.get('pr_auc', 0):.3f}**. "
                "Risk bands (<30% / 30–70% / >70%) are a simple thresholding of the predicted "
                "probability and are for demonstration only."
            )


def render_performance(metrics):
    st.subheader("Model comparison (5-fold stratified cross-validation)")
    st.write(
        f"Trained on **{metrics['dataset_rows']}** patient records "
        f"(~{metrics['positive_rate']:.0%} diabetic). All models share the same preprocessing "
        "(median imputation for missing clinical values, then standard scaling). Balanced class "
        "weights counter the imbalance, and **PR-AUC** is reported because it is more informative "
        "than ROC-AUC under class imbalance."
    )

    df = pd.DataFrame(metrics["models"])
    df["cv_roc_auc"] = df.apply(
        lambda r: f"{r['cv_roc_auc_mean']:.3f} ± {r['cv_roc_auc_std']:.3f}", axis=1
    )
    show = df[["model", "cv_roc_auc", "cv_accuracy_mean", "pr_auc", "roc_auc", "f1"]].rename(
        columns={"cv_accuracy_mean": "cv_accuracy", "roc_auc": "test_roc_auc", "f1": "test_f1"}
    )
    st.dataframe(
        show.style.format({"cv_accuracy_mean": "{:.3f}", "cv_accuracy": "{:.3f}",
                           "pr_auc": "{:.3f}", "test_roc_auc": "{:.3f}", "test_f1": "{:.3f}"}),
        width="stretch",
    )

    tuned = metrics.get("tuned_test_metrics", {})
    st.info(
        f"🏆 Best model: **{metrics['best_model']}** — tuned params `{metrics.get('best_params', {})}`. "
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
        s = pd.Series(fi).sort_values()
        fig = px.bar(
            x=s.values, y=s.index, orientation="h",
            labels={"x": "absolute weight", "y": ""},
            template="simple_white",
            color=s.values, color_continuous_scale="Blues",
        )
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Absolute model weight per feature (coefficients for linear models, impurity-based "
            "importance for trees)."
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
3. Train 4 classifiers (Logistic Regression, Random Forest, Gradient Boosting, KNN) with
   **5-fold stratified cross-validation**.
4. **Tune** the best model family with `GridSearchCV`.
5. Select by cross-validated ROC-AUC and save a reusable `sklearn` pipeline.

**Metrics** include accuracy, precision, recall, F1, ROC-AUC and **PR-AUC** (more informative
for the imbalanced, medical setting). Risk bands on the Predict tab are a simple thresholding
of the predicted probability for demonstration only.

**Tech stack:** Python, scikit-learn, pandas, Streamlit, Plotly.

*Disclaimer: This is a machine-learning demo for educational purposes and is **not** a medical
device. Do not use it for clinical diagnosis or treatment decisions.*
"""
    )


if __name__ == "__main__":
    main()
