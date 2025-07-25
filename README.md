# Pima Indians Diabetes Prediction

## Project Overview

This project aims to develop a machine learning model to predict the onset of diabetes in Pima Indian women. Utilizing a well-known dataset from the National Institute of Diabetes and Digestive and Kidney Diseases, this study focuses on leveraging diagnostic measurements to build a robust and interpretable predictive model.

The primary goal is to demonstrate a comprehensive machine learning workflow, from exploratory data analysis (EDA) and statistical testing to model building, evaluation, and interpretation, all within a structured and reproducible GitHub repository.

## Problem Statement

Diabetes is a chronic metabolic disease that affects millions worldwide. Early and accurate prediction can significantly improve patient outcomes through timely intervention and lifestyle modifications. This project addresses the challenge of predicting diabetes onset based on various health indicators, specifically within a population with a high prevalence of diabetes.

## Motivation

As an aspiring Machine Learning Engineer in healthcare data science, this project serves as a practical demonstration of key skills, including:

* **Data Understanding:** Ability to explore, visualize, and understand complex healthcare data.

* **Statistical Analysis:** Applying statistical tests to uncover relationships and validate assumptions.

* **Feature Engineering:** Creating meaningful features from raw data.

* **Model Development:** Building, training, and evaluating machine learning models.

* **Code Organization:** Structuring a machine learning project for reproducibility, maintainability, and collaboration.

* **Interpretation:** Explaining model predictions and insights to non-technical stakeholders.

## Dataset

The dataset used in this project is the **Pima Indians Diabetes Database**, originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It contains diagnostic measurements from 768 Pima Indian women, with the objective of predicting whether a patient has diabetes (1) or not (0).

**Key Features:**

* `Pregnancies`: Number of times pregnant

* `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test

* `BloodPressure`: Diastolic blood pressure (mm Hg)

* `SkinThickness`: Triceps skin fold thickness (mm)

* `Insulin`: 2-Hour serum insulin (mu U/ml)

* `BMI`: Body mass index (weight in kg/(height in m)^2)

* `DiabetesPedigreeFunction`: Diabetes pedigree function

* `Age`: Age (years)

* `Outcome`: Class variable (0 or 1) - 1 for diabetes, 0 for non-diabetes

**Data Source:** [Pima Indians Diabetes Database on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

**Note:** The raw dataset (`diabetes.csv`) is not directly included in this repository due to potential size considerations for larger projects, but can be easily downloaded from the Kaggle link above and placed into the `data/raw/` directory.

## Project Goals

1.  **Exploratory Data Analysis (EDA):**

    * Understand the distribution of each feature.

    * Identify potential outliers, missing values (represented as zeros in some columns), and inconsistencies.

    * Visualize relationships between features and the `Outcome` variable.

    * Analyze class imbalance in the `Outcome` variable.

2.  **Statistical Testing:**

    * Perform appropriate statistical tests (e.g., t-tests, chi-squared tests) to identify statistically significant differences between diabetic and non-diabetic groups for various features.

    * Assess correlations between features.

3.  **Data Preprocessing & Feature Engineering:**

    * Handle missing values (e.g., replacing zeros in `BloodPressure`, `BMI`, `Glucose` with appropriate imputation strategies).

    * Scale numerical features.

    * Potentially create new features (e.g., interaction terms, polynomial features) based on EDA and domain knowledge.

4.  **Model Development & Evaluation:**

    * Implement and train several classification models (e.g., Logistic Regression, Random Forest, Gradient Boosting models like XGBoost/LightGBM).

    * Utilize cross-validation for robust model evaluation.

    * Evaluate models using appropriate metrics for imbalanced datasets (e.g., Precision, Recall, F1-score, ROC AUC).

5.  **Model Interpretation:**

    * Analyze feature importance to understand which diagnostic measurements are most influential in predicting diabetes.

    * Potentially use techniques like SHAP or LIME for local model interpretability.

6.  **Reproducible Workflow:**

    * Maintain a clean, modular code structure.

    * Ensure all dependencies are listed in `requirements.txt`.

    * Provide clear instructions for running the project.

## Repository Structure

```
pima-diabetes-prediction/
├── data/
│   ├── raw/           # Original dataset (e.g., diabetes.csv)
│   └── processed/     # Cleaned and preprocessed data
├── notebooks/
│   ├── 1.0-EDA-Statistical_Analysis.ipynb  # Comprehensive EDA and statistical tests
│   └── 2.0-Model_Development_Evaluation.ipynb # Model training, evaluation, and interpretation
├── src/
│   ├── __init__.py    # Makes src a Python package
│   ├── data_loader.py # Functions for loading and initial cleaning of data
│   ├── preprocessor.py# Functions for data transformation and feature engineering
│   ├── models.py      # Functions for model training and prediction
│   └── train_pipeline.py # Main script to run the end-to-end ML pipeline
├── .gitignore         # Specifies intentionally untracked files to ignore
├── requirements.txt   # List of Python dependencies
└── README.md          # Project overview and documentation
```

## How to Run This Project

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BhagyashreeWaghale/pimaIndians-diabetes-prediction.git
    cd pima-diabetes-prediction
    ```
2.  **Download the dataset:**
    * Go to the [Pima Indians Diabetes Database on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
    * Click "Download" to get `diabetes.csv`.
    * Place the `diabetes.csv` file into the `data/raw/` directory.
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
5.  **Run the analysis and modeling:**
    * **For interactive exploration and detailed analysis:** Open and run the Jupyter notebooks in the `notebooks/` directory.
        
    * **To run the full end-to-end pipeline (after developing code in notebooks and refactoring to `src/`):**
        ```bash
        python src/train_pipeline.py
        ```
        

## Future Work / Improvements

* **Advanced Feature Engineering:** Explore more sophisticated feature creation, such as polynomial features or interaction terms, based on deeper domain knowledge or statistical insights.

* **Hyperparameter Tuning:** Implement more rigorous hyperparameter optimization techniques (e.g., GridSearchCV, RandomizedSearchCV, Optuna) for selected models.

* **Ensemble Methods:** Experiment with more complex ensemble techniques beyond simple voting classifiers, such as stacking or blending.

* **Deep Learning Models:** Explore the application of neural networks for this classification task.

* **Deployment:** Consider a simple deployment (e.g., using Flask/Streamlit) to demonstrate how the model could be used in a real-world application.

* **Fairness and Bias Analysis:** Investigate potential biases in the model's predictions across different demographic groups (if such information were available and ethically appropriate for this dataset).

* **CI/CD Pipeline:** Implement a basic CI/CD pipeline for automated testing and model retraining.

