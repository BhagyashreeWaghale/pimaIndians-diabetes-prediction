# EDA Observations and Insights Journal - Pima Indians Diabetes Prediction

This document serves as a log of key observations, statistical findings, and decisions made during the Exploratory Data Analysis (EDA) and initial data preprocessing phases of the Pima Indians Diabetes Prediction project.

## 1. Initial Data Loading and Overview
**Notebook: 1_EDA_Statistical_Analysis.ipynb**

* Dataset Size: 768 rows, 9 columns.

* Data Types: All features are numerical (integers or floats). The Outcome variable is binary (0 or 1).

* Descriptive Statistics: Initial inspection reveals a wide range of values across features.

## 2. Data Quality: Missing Values (Zeros) and Outliers
**Notebook: 1_EDA_Statistical_Analysis.ipynb**

* Explicit NaNs: No explicit NaN values found.

* Implicit Missing Values (Zeros):

* Glucose: 5 instances of 0.

* BloodPressure: 35 instances of 0.

* SkinThickness: 227 instances of 0.

* Insulin: 374 instances of 0.

* BMI: 11 instances of 0.

**Observation:** These 0 values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI are biologically impossible and indicate missing data.

**ML Approach Decision:** Direct row removal (listwise deletion) is not feasible due to the high count of zeros, especially in SkinThickness (~30% of data) and Insulin (~49% of data). This would significantly reduce data size and potentially introduce bias by altering the Outcome distribution. Therefore, imputation will be necessary.

## 3. Target Variable Distribution
**Notebook: 1_EDA_Statistical_Analysis.ipynb**

* Distribution:

    * Non-diabetic (0): 500 cases (~65.1%)

    * Diabetic (1): 268 cases (~34.9%)

**Observation:** The Outcome variable exhibits class imbalance, with a significantly higher proportion of non-diabetic cases.

**ML Approach Decision:** This imbalance is critical. Relying solely on accuracy as an evaluation metric would be misleading, as a model could achieve ~65% accuracy by simply predicting the majority class. Therefore, model evaluation will prioritize Precision, Recall, F1-score, and ROC AUC. Techniques such as oversampling (e.g., SMOTE), undersampling, or class weighting will be considered during model training to address this imbalance and improve the model's ability to predict the minority (diabetic) class.

## 4. Feature Distributions and Relationships (Visual & Quantitative)
**Notebook: 1_EDA_Statistical_Analysis.ipynb**

**Histograms (Visual Observation - Refer to image):**

* Pregnancies, DiabetesPedigreeFunction, Age: Visually appear right-skewed.

* Glucose, BloodPressure, BMI: Show some skew, but the presence of 0s distorts their true distribution.

* SkinThickness, Insulin: Heavily right-skewed, with a large spike at 0 (confirming missing values) and long tails indicating high values.

**Correlation Matrix (Visual Observation - Refer to image):**

* Glucose (0.47) and BMI (0.29) show the strongest positive Pearson correlation with Outcome, visually confirming their importance.

* Age and Pregnancies also show moderate positive correlation with Outcome.

* SkinThickness and Insulin are moderately correlated with each other (0.44), which is biologically plausible.

**Quantitative Skewness and Kurtosis (from results provided):**

**Skewness:**

* Highly right-skewed: Insulin (2.27), DiabetesPedigreeFunction (1.92), Age (1.13), Pregnancies (0.90).

* Negatively skewed: BloodPressure (-1.84), BMI (-0.43).

**Kurtosis:**   

* High kurtosis (heavy tails, more outliers): Insulin (7.21), DiabetesPedigreeFunction (5.59), BloodPressure (5.18), BMI (3.29).

**Observation:** Quantitative metrics confirm visual observations of non-normal distributions and heavy tails.

**ML Approach Decision:** Features with high absolute skewness (e.g., Insulin, DiabetesPedigreeFunction) will likely benefit from non-linear transformations (e.g., np.log1p for positive-only skewed data) to make their distributions more symmetrical. This can improve the performance of models sensitive to feature distribution (e.g., linear models). High kurtosis reinforces the need for robust handling of outliers.

## 5. Quantitative Outlier Analysis (IQR Method)
**Notebook: 1_EDA_Statistical_Analysis.ipynb**

* Outlier Counts (excluding 0s for relevant columns, from results provided):

    * Pregnancies: 4 outliers

    * Glucose: 0 outliers (after excluding 0s, values are within typical ranges)

    * BloodPressure: 10 outliers (after excluding 0s)

    * SkinThickness: 1 outlier (after excluding 0s)

    * Insulin: 34 outliers (after excluding 0s)

    * BMI: 8 outliers (after excluding 0s)

    * DiabetesPedigreeFunction: 29 outliers

    * Age: 9 outliers

**Observation:** Analysis confirms numerous outliers, particularly in Insulin and DiabetesPedigreeFunction. While SkinThickness only shows 1 outlier after excluding zeros, its high number of zeros and extreme skewness still make it challenging.

**ML Approach Decision:** The significant presence of outliers suggests that a RobustScaler might be more appropriate for feature scaling than StandardScaler or MinMaxScaler, as it is less sensitive to extreme values by using medians and IQRs. Further investigation into capping or winsorization for highly influential outliers might be considered.

## 6. Statistical Test Results: Feature Importance & Distribution Differences
**Notebook: 1_EDA_Statistical_Analysis.ipynb**

**Shapiro-Wilk Test for Normality:**

P-values for ALL features are 0.000 (p < 0.05).

**Observation:** This quantitatively confirms that none of the features are normally distributed. This is a crucial finding, validating visual observations and skewness/kurtosis metrics.

**ML Approach Decision:** Since normality assumptions are violated, non-parametric statistical tests are more appropriate for comparing group differences. This also reinforces the need for feature transformations and/or using tree-based models (like Random Forests, XGBoost) which are less sensitive to feature distribution assumptions.

**T-tests (Independent Samples) for Numerical Features vs. Outcome:**

* Significant Differences (p < 0.05): Pregnancies (p=0.000), Glucose (p=0.000), SkinThickness (p=0.049), Insulin (p=0.001), BMI (p=0.000), DiabetesPedigreeFunction (p=0.000), Age (p=0.000).

    **Interpretation:** The means of these features are statistically different between the diabetic and non-diabetic groups, suggesting their predictive importance.

* No Significant Difference (p >= 0.05): BloodPressure (p=0.087).

    **Interpretation:** Based on the t-test, the mean blood pressure between groups is not statistically different. However, given the non-normality and zeros, the t-test's reliability here is questionable.

**Mann-Whitney U Test (Non-parametric) for Numerical Features vs. Outcome:**

Significant Differences (p < 0.05) for ALL features: Pregnancies (p=0.000), Glucose (p=0.000), BloodPressure (p=0.000), SkinThickness (p=0.000), Insulin (p=0.000), BMI (p=0.000), DiabetesPedigreeFunction (p=0.000), Age (p=0.000).

**Interpretation:** This is the most robust and critical finding. Given the confirmed non-normality, the Mann-Whitney U test's results are more reliable than the t-test. It indicates that the distributions of ALL features are statistically different between the diabetic and non-diabetic groups. This strongly suggests that every feature in the dataset holds predictive power for the Outcome.

**ML Approach Decision:** The Mann-Whitney U test results solidify the decision to use all features for modeling, as they all show a statistically significant relationship with the target variable, even BloodPressure.


## 7. Comprehensive Preprocessing & Modeling Strategy
Based on these detailed observations and statistical validations, the refined ML approach will be:

**Imputation:**

Replace 0s in Glucose, BloodPressure, BMI, SkinThickness, and Insulin with the median of their respective columns. Median is chosen for robustness against skewness and outliers.

**Transformation:**

* Apply np.log1p transformation to highly skewed features (Insulin, DiabetesPedigreeFunction, Pregnancies, Age, SkinThickness after imputation) to normalize their distributions.

* Outlier Handling: While scaling will help, further investigation into capping or winsorization for extreme outliers in features like Insulin might be beneficial if models struggle.

* Feature Scaling: Apply StandardScaler to all numerical features after imputation and transformation. RobustScaler can be tried out as an alternative.

* Class Imbalance: Implement strategies such as SMOTE (Synthetic Minority Over-sampling Technique) on the training data or use class weights within the chosen machine learning model to address the imbalance in the Outcome variable.

**Model Selection:**

* Prioritize robust, non-parametric models like Random Forest and Gradient Boosting (XGBoost/LightGBM), as they are less sensitive to feature distributions and outliers.

* Include Logistic Regression as a baseline, acknowledging its sensitivity to non-normal data and outliers.

**Model Evaluation:** Focus on F1-score, Precision, Recall, and ROC AUC for comprehensive evaluation, especially for the minority class.

**Model Interpretation:** Prioritize understanding feature importance and model decisions (e.g., using SHAP values), given the healthcare context.


## 8. Iterative Preprocessing: The Impact of Imputation, Transformation, and Scaling

Following the initial EDA, a multi-step preprocessing pipeline was implemented to address data quality issues and prepare the features for machine learning models. The effects of each step were quantitatively and visually analyzed.

**Imputation (Median):**

Replacing the biologically impossible zeros with the median significantly altered the distributions of several key features.

| Feature | Original Skewness | Imputed Skewness | Skewness Change | 
|---|---|---|---|
| `Glucose` | 0.17 | 0.54 | +0.37 | 
| `BloodPressure` | \-1.84 | 0.14 | +1.98 | 
| `SkinThickness` | 0.11 | 0.84 | +0.73 | 
| `Insulin` | 2.27 | 3.38 | +1.11 | 
| `BMI` | \-0.43 | 0.60 | +1.03 | 

* Insights: Imputation corrected the deceptive symmetry caused by the spike at zero, revealing the true right-skew of features like Insulin and SkinThickness. The large change in BloodPressure's skewness (from -1.84 to 0.14) shows the successful mitigation of its zero-value problem.
  

**Transformation (log1p):**

The np.log1p transformation was applied to highly skewed features to normalize their distributions.

| Feature | Imputed Skewness | Transformed Skewness | Skewness Change |
| :--- | :--- | :--- | :--- |
| `Pregnancies` | 0.90 | -0.24 | -1.14 |
| `Glucose` | 0.54 | -0.06 | -0.60 |
| `SkinThickness` | 0.84 | -0.86 | -1.70 |
| `Insulin` | 3.38 | -0.18 | -3.56 |
| `BMI` | 0.60 | -0.04 | -0.64 |

* Insights: 
    * The transformation was extremely effective, particularly for the most problematic feature, Insulin, reducing its skewness from 3.38 to a near-symmetrical -0.18. Similarly, SkinThickness saw a significant shift, demonstrating the power of transformations to prepare data for modeling. The changes in skewness are consistently moving towards a value of zero, indicating a more normal distribution.
    * The change in kurtosis values reveals the true effectiveness of the log1p transformation in handling the data's heavy tails. The kurtosis of Insulin, for instance, was reduced from an extremely high 16.23 after imputation to a more manageable 3.31 after transformation. This demonstrates that preprocessing has successfully prepared even the most challenging features for robust modeling.


**Scaling (RobustScaler) - Skewness**

Even after log1p transformation, features with extremely high kurtosis (like Insulin and SkinThickness) still have influential outliers. This makes RobustScaler a very strong candidate for scaling, as it's designed to handle such distributions without being unduly influenced by outliers. As scaling does not change the shape of the distribution, the skewness values remain the same as the transformed data.

| Feature | Transformed Skewness | Scaled Skewness | Skewness Change |
| :--- | :--- | :--- | :--- |
| `Pregnancies` | -0.24 | -0.24 | 0.00 |
| `Glucose` | -0.06 | -0.06 | 0.00 |
| `BloodPressure` | 0.14 | 0.14 | 0.00 |
| `SkinThickness` | -0.86 | -0.86 | 0.00 |
| `Insulin` | -0.18 | -0.18 | 0.00 |
| `BMI` | -0.04 | -0.04 | 0.00 |
| `DiabetesPedigreeFunction` | 1.12 | 1.12 | 0.00 |
| `Age` | 0.61 | 0.61 | 0.00 |

* Insights: This table confirms that scaling works as intended by standardizing the data's range without altering the shape of the distribution. The fact that the skewness values remain at or near zero after this step is a success, as a symmetrical distribution is ideal for many models
