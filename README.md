[README.md](https://github.com/user-attachments/files/30885994/README.md)
# Titanic Survival Prediction

A machine learning classification project that predicts passenger survival on the RMS Titanic, and explores what factors made survival more likely.

## Problem Statement

While survival on the Titanic involved an element of luck, certain groups — women, children, and higher-class passengers — had a much better chance of survival. This project builds and compares two classification models to answer: **what sorts of people were more likely to survive?**

## Dataset

[Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data) (Kaggle), 891 passenger records with features including age, sex, ticket class, fare, and family relationships.

## Approach

- **EDA**: explored survival rates by sex, class, and port of embarkation
- **Data cleaning**: imputed missing `Age` (median by class/sex group) and `Embarked` (mode); converted sparse `Cabin` into a `HasCabin` indicator
- **Feature engineering**: added `FamilySize` and `IsAlone`
- **Models**: Logistic Regression (baseline, interpretable) vs. Random Forest (captures non-linear interactions)
- **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, and 5-fold cross-validation

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.774 | 0.696 | 0.733 | 0.845 |
| Random Forest | 0.793 | 0.767 | 0.667 | 0.713 | 0.843 |

**Key finding**: Sex was by far the strongest predictor of survival, followed by fare/passenger class and age — consistent with the historical "women and children first" evacuation priority.

## Files

- `Titanic_Survival_Prediction.ipynb` — full analysis notebook (EDA, cleaning, feature engineering, model training, evaluation)
- `Titanic_Survival_Report.docx` — written report with methodology, results, and discussion
- `Titanic-Dataset.csv` — dataset used

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

Open `Titanic_Survival_Prediction.ipynb` in Jupyter Notebook or JupyterLab and run all cells.
