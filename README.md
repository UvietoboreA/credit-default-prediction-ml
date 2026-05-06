# Credit Default Prediction Using Machine Learning

## Project Overview

This project applies machine learning techniques to predict whether a customer is likely to default on credit payments based on financial and demographic information.

The project includes data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and comparison of multiple machine learning algorithms.

The main objective is to support financial institutions in identifying high-risk customers, improving lending decisions, and reducing financial losses caused by defaults.

---

## Dataset

The dataset contains customer financial and demographic attributes used to predict credit default behaviour.

Example features include:

- Credit limit
- Age
- Gender
- Education
- Marital status
- Bill statement history
- Previous payment history
- Payment amounts

Target variable:

- `default`
  - `1` = Customer defaults
  - `0` = Customer does not default

The dataset was split into training and testing datasets:

- `creditdefault_train.csv`
- `creditdefault_test.csv`

---

## Project Objectives

- Clean and preprocess financial data
- Handle missing values and categorical variables
- Perform exploratory data analysis (EDA)
- Identify important risk-related features
- Train and compare classification models
- Evaluate predictive performance
- Analyse the impact of class imbalance
- Improve risk prediction accuracy

---

## Exploratory Data Analysis

The project includes:

- Default class distribution analysis
- Correlation analysis
- Payment history analysis
- Credit limit distribution analysis
- Financial behaviour visualisations
- Heatmaps and statistical summaries
- Feature relationship exploration

---

## Machine Learning Models Used

The following machine learning models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbours (KNN)
- XGBoost Classifier

---

## Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score
- Confusion Matrix

---

## Why Precision Is Important in Credit Risk

In credit default prediction, precision is highly important because false positives can negatively affect business decisions.

A false positive occurs when the model predicts that a customer will default even though they would not actually default. This may cause financial institutions to reject reliable customers unnecessarily.

Balancing precision and recall is therefore critical in financial risk modelling.

---

## Technologies Used

- R Programming
- Machine Learning Algorithms
- Data Visualisation
- Statistical Analysis
- CSV Data Processing

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/credit-default-prediction-ml.git
```

### 2. Open the project directory

```bash
cd credit-default-prediction-ml
```

### 3. Open the R script

Open:

```text
Joshua_Coursework_2.R
```

in RStudio or another R environment.

### 4. Install required packages

```r
install.packages(c("caret", "randomForest", "xgboost", "ggplot2", "dplyr"))
```

### 5. Run the script

Execute the script from top to bottom.

---

## Repository Structure

```text
.
├── Joshua_Coursework_2.R
├── creditdefault_train.csv
├── creditdefault_test.csv
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Key Challenges

- Handling class imbalance in credit default data
- Managing financial feature correlations
- Reducing false positive predictions
- Improving model generalisation on unseen data

---

## Future Improvements

- Apply SMOTE for imbalance handling
- Perform advanced hyperparameter tuning
- Use ensemble learning methods
- Add explainable AI techniques such as SHAP
- Deploy the model as a financial risk dashboard
- Compare deep learning approaches

---

## Repository Description

Machine learning project for predicting customer credit default risk using financial and demographic data analysis.

---

## .gitignore

```text
.Rhistory
.RData
.Rproj.user
.DS_Store
```

---


## Author

Uvietobore Joshua Adjugah

MSc Data Science and Artificial Intelligence
