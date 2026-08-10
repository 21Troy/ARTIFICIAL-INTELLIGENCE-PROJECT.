[README.md](https://github.com/user-attachments/files/30908867/README.md)
# Titanic Survival Prediction

## Machine Learning Course Project

This project uses the classic Titanic passenger dataset to investigate
the question:

> **What sorts of people were more likely to survive the Titanic
> disaster?**

The project treats Titanic survival prediction as a **binary
classification problem**, where:

-   `Survived = 0` → Did not survive
-   `Survived = 1` → Survived

The analysis uses passenger characteristics such as age, sex, passenger
class, fare, and family relationships to explore survival patterns and
build predictive machine-learning models.

------------------------------------------------------------------------

## Project Objectives

The main objectives of this project are to:

1.  Explore and understand the Titanic passenger dataset.
2.  Identify and handle missing data.
3.  Engineer useful features for prediction.
4.  Explore relationships between passenger characteristics and
    survival.
5.  Train multiple classification models.
6.  Compare model performance using several evaluation metrics.
7.  Identify the features that have the greatest influence on survival.
8.  Perform statistical significance testing and correlation analysis.
9.  Tune model hyperparameters using cross-validation.
10. Analyze model errors and discuss the limitations of the prediction
    task.

------------------------------------------------------------------------

## Dataset

The project uses the **Titanic - Machine Learning from Disaster**
dataset from Kaggle.

Dataset source:

https://www.kaggle.com/c/titanic/data

The notebook expects the dataset file to be named:

``` text
Titanic-Dataset.csv
```

and located in the same directory as the Jupyter Notebook.

### Main Features Used

Some of the important variables in the dataset include:

  Feature      Description
  ------------ -----------------------------------------
  `Survived`   Target variable: 0 = died, 1 = survived
  `Pclass`     Passenger class
  `Sex`        Passenger sex
  `Age`        Passenger age
  `SibSp`      Number of siblings/spouses aboard
  `Parch`      Number of parents/children aboard
  `Fare`       Passenger fare
  `Embarked`   Port of embarkation
  `Cabin`      Cabin information

------------------------------------------------------------------------

## Technologies and Libraries

The project was developed in **Python using Jupyter Notebook**.

### Main libraries

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   SciPy
-   Jupyter Notebook

### Machine Learning Algorithms

Three classification algorithms are evaluated:

1.  **Logistic Regression**
2.  **Random Forest**
3.  **Gradient Boosting**

------------------------------------------------------------------------

## Project Workflow

The notebook follows the following machine-learning workflow:

### 1. Data Loading

The Titanic dataset is loaded into a Pandas DataFrame from:

``` python
pd.read_csv("Titanic-Dataset.csv")
```

The dataset is then inspected using functions such as:

``` python
df.head()
df.info()
df.describe()
```

------------------------------------------------------------------------

### 2. Exploratory Data Analysis

The project investigates:

-   Missing values
-   Overall survival rate
-   Survival by sex
-   Survival by passenger class
-   Survival by port of embarkation
-   Age distribution and survival
-   Family size and survival

The analysis shows strong differences in survival across several
passenger characteristics.

------------------------------------------------------------------------

### 3. Data Cleaning

The following preprocessing steps are performed:

-   Missing `Age` values are filled using the median age within each
    `Pclass` and `Sex` group.
-   Missing `Embarked` values are filled using the mode.
-   `Cabin` is converted into a binary `HasCabin` feature because a
    large proportion of cabin values are missing.
-   `PassengerId`, `Ticket`, `Cabin`, and `Name` are removed from the
    raw modeling features.

------------------------------------------------------------------------

### 4. Feature Engineering

Two additional features are created:

#### FamilySize

``` text
FamilySize = SibSp + Parch + 1
```

This represents the total number of family members traveling together,
including the passenger.

#### IsAlone

A binary feature indicating whether the passenger was traveling alone:

-   `1` → Passenger traveled alone
-   `0` → Passenger was not alone

Categorical variables such as `Sex` and `Embarked` are converted into
numerical variables using one-hot encoding.

------------------------------------------------------------------------

### 5. Train/Test Split

The dataset is divided into:

-   **80% training data**
-   **20% test data**

A stratified split is used so that the distribution of the target
variable is maintained between the training and test sets.

The random state is set to:

``` python
RANDOM_STATE = 42
```

------------------------------------------------------------------------

### 6. Feature Scaling

Numeric features used by Logistic Regression are standardized using
`StandardScaler`.

The scaled features include:

-   `Age`
-   `Fare`
-   `FamilySize`
-   `SibSp`
-   `Parch`

Tree-based models such as Random Forest and Gradient Boosting are
trained without scaling.

------------------------------------------------------------------------

## Model Training

### Logistic Regression

Logistic Regression is used as an interpretable linear baseline model.

It provides coefficients that help explain how different features affect
the predicted probability of survival.

### Random Forest

Random Forest is an ensemble of decision trees. It is capable of
modeling non-linear relationships and interactions between features.

### Gradient Boosting

Gradient Boosting is also evaluated as a third model. Its
hyperparameters are tuned using cross-validation to improve predictive
performance.

------------------------------------------------------------------------

## Model Evaluation

The models are evaluated using several metrics:

-   **Accuracy**
-   **Precision**
-   **Recall**
-   **F1-Score**
-   **ROC-AUC**
-   **PR-AUC**

The project also uses:

-   Confusion matrices
-   ROC curves
-   Precision-Recall curves
-   5-fold cross-validation
-   Learning curves

These evaluation methods provide a broader view of model performance
rather than relying only on accuracy.

------------------------------------------------------------------------

## Hyperparameter Tuning

`GridSearchCV` with **5-fold stratified cross-validation** is used to
tune the main hyperparameters of the three models.

The tuning process evaluates different parameter combinations and
selects the configurations that provide the best cross-validated
accuracy on the training data.

------------------------------------------------------------------------

## Statistical Analysis

The project also tests whether observed relationships with survival are
statistically significant.

### Chi-Square Tests

Chi-square tests are performed for categorical variables:

-   `Sex`
-   `Pclass`
-   `Embarked`

### Independent T-Tests

Independent t-tests are performed for continuous variables:

-   `Age`
-   `Fare`

The notebook reports p-values below the conventional `0.05` significance
threshold for the tested relationships.

------------------------------------------------------------------------

## Feature Importance

The project examines which variables contribute most strongly to
survival prediction.

Two approaches are used:

### Random Forest Feature Importance

Random Forest feature importance is used to rank the predictive
features.

### Logistic Regression Coefficients

Logistic Regression coefficients are examined to determine the direction
of the relationship:

-   Positive coefficient → increases predicted survival odds.
-   Negative coefficient → decreases predicted survival odds.

------------------------------------------------------------------------

## Key Findings

The analysis identifies several important survival patterns.

### Sex

Sex is identified as the strongest predictor of survival. Female
passengers had substantially higher survival odds than male passengers.

### Passenger Class and Fare

Passenger class and fare are strongly related to survival. Higher-class
and higher-fare passengers generally had better survival outcomes.

### Age

Age is associated with survival, with younger passengers, particularly
children, generally showing better survival outcomes.

### Cabin Information

Having recorded cabin information (`HasCabin`) provides a secondary
predictive signal and can act as a proxy for passenger
accommodation/class.

### Family Size

Family relationships also contribute to survival prediction. Small
family groups generally showed more favorable outcomes than passengers
traveling alone or in very large groups.

------------------------------------------------------------------------

## Model Performance

According to the notebook's final analysis:

-   The three tuned models achieved approximately **78--80% test-set
    accuracy**.
-   Test-set ROC-AUC values were approximately **0.80--0.84**.
-   Logistic Regression achieved the best reported test-set ROC-AUC of
    approximately **0.845**.
-   Gradient Boosting achieved the best reported cross-validated
    accuracy of approximately **83.3%** and strong precision.

Based on the notebook's interpretation, **Logistic Regression is
recommended as the primary model because of its strong performance and
interpretability**, while **Gradient Boosting** is considered a strong
alternative when accuracy is the main priority.

------------------------------------------------------------------------

## Error Analysis

The project examines passengers that the Random Forest model incorrectly
classified.

The analysis distinguishes between:

-   **False negatives** --- passengers who survived but were predicted
    to have died.
-   **False positives** --- passengers who died but were predicted to
    have survived.

The notebook reports that many missed survivors were lower-fare,
third-class passengers who survived despite characteristics associated
with lower survival probability.

Many missed deaths were higher-fare passengers who died despite having
characteristics generally associated with better survival outcomes.

This demonstrates that the models capture broad population-level
patterns but cannot account for every individual circumstance.

------------------------------------------------------------------------

## Limitations

The project identifies several limitations:

1.  The dataset contains only **891 labeled records**, limiting the
    amount of information available for training.
2.  Approximately **20% of Age values** and more than **77% of Cabin
    values** were missing.
3.  Missing cabin information required simplification into the
    `HasCabin` indicator.
4.  The available variables cannot capture all individual circumstances
    that influenced survival.
5.  Even after hyperparameter tuning, model performance remains below
    perfect accuracy because survival depended on factors that are not
    represented in the dataset.

------------------------------------------------------------------------

## Future Improvements

The notebook recommends several possible improvements:

-   Extract passenger titles such as `Mr.`, `Mrs.`, `Miss.`, and
    `Master` from `Name`.
-   Explore stacking or other ensemble approaches combining the three
    models.
-   Add more detailed cabin information, such as cabin deck/location.
-   Investigate additional features that may better represent
    passengers' physical proximity to lifeboats.
-   Evaluate additional machine-learning algorithms and
    feature-engineering techniques.

------------------------------------------------------------------------

## Project Structure

The GitHub repository structure is:

``` text
Titanic-Survival-Prediction/
│
├── Titanic_Survival_Prediction-1.ipynb
├── Titanic-Dataset.csv
├── README.md
└── .gitignore
```

The notebook contains the complete analysis, preprocessing, feature
engineering, model training, evaluation, statistical testing,
hyperparameter tuning, and conclusions.

------------------------------------------------------------------------

## How to Run the Project

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd Titanic-Survival-Prediction
```

### 2. Install the required libraries

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### 3. Start Jupyter Notebook

``` bash
jupyter notebook
```

### 4. Open the notebook

Open:

``` text
Titanic_Survival_Prediction-1.ipynb
```

### 5. Ensure the dataset is available

Place:

``` text
Titanic-Dataset.csv
```

in the same directory as the notebook.

### 6. Run the notebook

Run the cells from top to bottom to reproduce the analysis.

------------------------------------------------------------------------

## Conclusion

This project demonstrates an end-to-end machine-learning workflow for
predicting Titanic passenger survival.

The analysis combines exploratory data analysis, data cleaning, feature
engineering, statistical testing, classification, hyperparameter tuning,
model evaluation, feature interpretation, learning-curve analysis, and
error analysis.

The results indicate that **sex, passenger class, fare, age, cabin
information, and family characteristics** are important factors
associated with Titanic survival.

The project also demonstrates why model interpretability and multiple
evaluation methods are important when selecting a machine-learning
model.

------------------------------------------------------------------------

## Author

**Titanic Survival Prediction --- Machine Learning Course Project**

Developed using Python and Jupyter Notebook.
