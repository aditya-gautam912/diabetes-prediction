# Diabetes Prediction Using Machine Learning 🩺

A machine-learning web app that predicts the likelihood of diabetes from clinical
measurements, built on the classic **Pima Indians Diabetes dataset**.

It trains several classifiers, selects the best by ROC-AUC, and serves an
interactive **Streamlit** app where you can enter patient data and get an instant
risk prediction plus a view of model performance.

## Demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL (default `http://localhost:8501`).

## Features

- **Interactive prediction** — adjust sliders for the 8 clinical features and get a
  diabetic / not-diabetic prediction with a calibrated probability.
- **Model comparison** — accuracy, precision, recall, F1 and ROC-AUC for four models.
- **Transparent preprocessing** — clinical `0`s are treated as *missing* and
  median-imputed, then features are standardized, all inside a single saved
  `sklearn` pipeline so the app predicts directly from raw input.
- **Auto-trains on first run** if the saved model is absent (e.g. on Streamlit Cloud).

## Machine-learning pipeline

1. **Data:** Pima Indians Diabetes dataset (768 patients, 8 features, ~35% positive).
2. **Cleaning:** replace `0` with `NaN` in `Glucose, BloodPressure, SkinThickness,
   Insulin, BMI` (0 is not a valid clinical value there).
3. **Preprocess:** median imputation → `StandardScaler`.
4. **Models:** Logistic Regression, Random Forest, Gradient Boosting, K-Nearest Neighbors.
5. **Selection:** best model by ROC-AUC is refit on all data and saved to
   `models/best_model.joblib`.

## Results (held-out 20% test set, `random_state=42`)

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|----------|-----------|--------|-------|---------|
| **Gradient Boosting** | **0.779** | **0.717** | 0.611 | **0.660** | **0.832** |
| Logistic Regression | 0.695    | 0.575     | 0.500  | 0.535 | 0.812   |
| Random Forest       | 0.734    | 0.638     | 0.556  | 0.594 | 0.810   |
| K-Nearest Neighbors | 0.721    | 0.628     | 0.500  | 0.557 | 0.762   |

**Best model: Gradient Boosting (ROC-AUC 0.832).**

## Project structure

```
diabetes-prediction/
├── app.py                # Streamlit app (loads the saved model)
├── src/
│   └── train.py          # Data cleaning, training, evaluation, saving
├── data/
│   └── diabetes.csv      # Pima Indians dataset (with headers)
├── models/
│   ├── best_model.joblib # Saved sklearn pipeline
│   └── metrics.json      # Per-model metrics report
├── requirements.txt
└── README.md
```

## Tech stack

Python · scikit-learn · pandas · NumPy · joblib · Streamlit

## Disclaimer

This is an educational machine-learning demo, **not a medical device**. It must
not be used for clinical diagnosis or treatment decisions.
