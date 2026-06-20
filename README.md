# Diabetes Risk Screening

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

An end-to-end machine learning project: exploratory data analysis and model training on the Pima Indians Diabetes dataset, followed by a Streamlit app that uses the trained model to estimate diabetes risk from patient measurements.

## Table of contents

- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Model training](#model-training)
- [Results](#results)
- [Running the app](#running-the-app)
- [Requirements](#requirements)
- [Notes](#notes)

## Project structure

```
.
├── app.py                                          # Streamlit prediction app
├── diabetes_model.pkl                              # trained Logistic Regression model
├── scaler.pkl                                       # fitted StandardScaler
├── diabetes.csv                                      # dataset
├── notebooks/
│   └── diabetes-eda-and-model-training.ipynb        # EDA, cleaning, model comparison, export
├── requirements.txt
└── README.md
```

## Dataset

The [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) contains 768 patient records with eight clinical measurements (`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`) and a binary target, `Outcome` (1 = diabetic, 0 = not diabetic).

## Model training

The notebook in `notebooks/diabetes-eda-and-model-training.ipynb` walks through:

- Exploratory analysis of feature distributions, class balance, and correlations
- Outlier handling on `Age` using the IQR rule
- Treating biologically impossible zero-values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` as missing data, imputed with the column median
- Comparing Logistic Regression (baseline and class-weight-balanced), KNN, Decision Tree, and Random Forest
- Exporting the final model and scaler to `diabetes_model.pkl` / `scaler.pkl`

## Results

Metrics on a held-out 20% test split:

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.763 | 0.609 | 0.847 |
| Logistic Regression (balanced) | 0.750 | 0.667 | 0.845 |
| KNN | 0.730 | 0.586 | 0.775 |
| Decision Tree | 0.730 | 0.586 | 0.687 |
| Random Forest | 0.724 | 0.580 | 0.823 |

The baseline Logistic Regression was deployed in the app — it had the strongest ROC-AUC and stays interpretable. The class-weight-balanced variant trades some precision for higher recall on diabetic cases, which may be preferable if missing a true positive is the bigger concern for your use case; see the notebook for that comparison.

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `diabetes_model.pkl` and `scaler.pkl` from the same folder, so keep them alongside `app.py` (or update the paths in `app.py` if you move them).

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
plotly
```

Install with:

```bash
pip install -r requirements.txt
```

## Notes

This is an educational project, not a diagnostic tool. Predictions are estimates based on a small, public dataset and should not be used for real medical decisions — please consult a healthcare professional for medical advice. Feedback and suggestions are welcome via issues or pull requests.



## Live Application
👉 https://diabetes-risk-prediction-y6gvlxnbzanjgev7kagpgm.streamlit.app/
