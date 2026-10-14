# Diabetes Prediction Using Machine Learning 🩺

**▶ [Live demo](https://diabetes-prediction-gh7vmdsdp9wrpxd6e4vudv.streamlit.app/)** ·
[GitHub repo](https://github.com/aditya-gautam912/diabetes-prediction)

A machine-learning web app that predicts the likelihood of diabetes from clinical
measurements, built on the classic **Pima Indians Diabetes dataset**.

It trains several classifiers with **cross-validation**, **tunes** the best one,
and serves an interactive **Streamlit** app where you can enter patient data and
get an instant risk probability plus a full performance view (ROC curve, confusion
matrix, feature importance).

## Demo

Try the **[live app](https://diabetes-prediction-gh7vmdsdp9wrpxd6e4vudv.streamlit.app/)**
(no setup needed), or run it locally:

```bash
pip install -r requirements.txt
python src/train.py        # (re)train and save model + plots
streamlit run app.py
```

Then open the local URL (default `http://localhost:8501`).

## Features

- **Interactive prediction** — adjust sliders for the 8 clinical features and get a
  diabetic / not-diabetic prediction with a calibrated probability.
- **Model comparison** — cross-validated accuracy & ROC-AUC (mean ± std), plus
  test-set PR-AUC, ROC-AUC and F1 for four models.
- **Tuned best model** — `GridSearchCV` over the winning model family.
- **Interpretability** — feature-importance chart, ROC curve and confusion-matrix heatmap.
- **Transparent preprocessing** — clinical `0`s are treated as *missing* and
  median-imputed, then standardized, all inside a single saved `sklearn` pipeline.
- **Auto-trains on first run** if the saved model is absent (e.g. on Streamlit Cloud).

## Machine-learning pipeline

1. **Data:** Pima Indians Diabetes dataset (768 patients, 8 features, ~35% positive).
2. **Cleaning:** replace `0` with `NaN` in `Glucose, BloodPressure, SkinThickness,
   Insulin, BMI` (0 is not a valid clinical value there).
3. **Preprocess:** median imputation → `StandardScaler`.
4. **Models:** Logistic Regression, Random Forest, Gradient Boosting, K-Nearest
   Neighbors — evaluated with **5-fold stratified cross-validation**. Balanced class
   weights counter the class imbalance; **PR-AUC** is reported because it is more
   informative than ROC-AUC under imbalance.
5. **Tuning:** `GridSearchCV` on the winning model family.
6. **Selection:** best model by cross-validated ROC-AUC is refit on all data and
   saved to `models/best_model.joblib`, along with ROC and confusion-matrix plots.

## Results

**Cross-validation (5-fold, stratified):**

| Model               | CV ROC-AUC (mean±std) | PR-AUC | Test ROC-AUC | Test F1 |
|---------------------|------------------------|--------|--------------|---------|
| Logistic Regression | **0.836 ± 0.021**      | 0.671  | 0.812        | 0.535   |
| Random Forest       | 0.830 ± 0.020          | 0.693  | 0.810        | 0.594   |
| Gradient Boosting   | 0.827 ± 0.029          | **0.719** | 0.832    | **0.660** |
| K-Nearest Neighbors | 0.747 ± 0.026          | 0.567  | 0.762        | 0.557   |

**Tuned best model: Logistic Regression (`C=10.0`) — CV ROC-AUC 0.844.**
Held-out test set: ROC-AUC **0.812**, PR-AUC **0.672**, accuracy **0.734**, F1 **0.650**.

**Top features** (absolute standardized coefficients): Glucose, Diabetes
Pedigree Function, BMI.

## Testing & CI

- `tests/test_model.py` — 5 unit tests (artifact presence, metrics structure,
  probability range, risk ordering, binary output).
- `.github/workflows/ci.yml` — GitHub Actions pipeline: install → retrain
  (reproducibility check) → `pytest` on every push/PR to `main`.

```bash
pytest -q
```

## Project structure

```
diabetes-prediction/
├── app.py                     # Streamlit app (loads the saved model)
├── src/
│   └── train.py               # Cleaning, CV, tuning, evaluation, plots, saving
├── data/
│   └── diabetes.csv           # Pima Indians dataset (with headers)
├── models/
│   ├── best_model.joblib      # Tuned sklearn pipeline
│   ├── metrics.json           # CV + test metrics, feature importances
│   ├── roc_curve.png          # ROC curve of the tuned model
│   └── confusion_matrix.png   # Confusion-matrix heatmap
├── tests/
│   └── test_model.py          # Unit tests
├── .github/workflows/ci.yml   # CI
├── requirements.txt
└── README.md
```

## Tech stack

Python · scikit-learn · pandas · NumPy · joblib · matplotlib · seaborn · Streamlit · pytest

## Disclaimer

This is an educational machine-learning demo, **not a medical device**. It must
not be used for clinical diagnosis or treatment decisions.
