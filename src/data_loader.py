import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_and_clean_data(file_path):
    """
    Loads the Pima Indians Diabetes dataset and performs initial cleaning
    by replacing '0' values in specific columns with NaN, as these represent
    missing or biologically impossible measurements.

    Args:
        file_path (str): The path to the raw diabetes.csv file.

    Returns:
        pandas.DataFrame: The loaded DataFrame with initial missing values handled.
    """
    df = pd.read_csv(file_path)

    # Define columns where 0s represent missing/invalid data
    # These were identified during EDA
    cols_with_zeros_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Replace 0s with NaN in these specific columns
    for col in cols_with_zeros_as_missing:
        df[col] = df[col].replace(0, np.nan)

    print(f"Data loaded from {file_path} and 0s replaced with NaN in {cols_with_zeros_as_missing}.")
    print("Missing values after initial cleaning:")
    print(df.isnull().sum())

    return df

def impute_missing_values(df):
    """
    Imputes missing values (NaNs) in the DataFrame using median imputation
    for specific columns based on EDA insights.

    Args:
        df (pandas.DataFrame): The DataFrame with NaNs from load_and_clean_data.

    Returns:
        pandas.DataFrame: The DataFrame with missing values imputed.
    """
    df_imputed = df.copy() # Work on a copy to avoid modifying the original DataFrame

    # Imputation strategy based on EDA insights (median is robust to skewness/outliers)
    # Median for Glucose, BloodPressure, BMI (robust to remaining outliers)
    # Median for SkinThickness, Insulin (highly skewed, many zeros)
    imputation_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    imputer = SimpleImputer(strategy='median')
    df_imputed[imputation_cols] = imputer.fit_transform(df_imputed[imputation_cols])
    
    print("\nMissing values after median imputation:")
    print(df_imputed.isnull().sum()) 

    return df_imputed

# Example usage (for testing this script independently)
if __name__ == "__main__":
    # Assuming the script is run from the project root or src directory
    # Adjust path if running from a different location for testing
    raw_data_path = '../data/raw/diabetes.csv'

    # Load and initially clean the data
    cleaned_df = load_and_clean_data(raw_data_path)

    # Impute the missing values
    imputed_df = impute_missing_values(cleaned_df)

    print("\nFirst 5 rows of the fully processed data:")
    print(imputed_df.head())

    # Optionally, save the processed data
    imputed_data_path = '../data/processed/diabetes_imputed.csv'
    imputed_df.to_csv(imputed_data_path, index=False)
    print(f"Imputed data saved to: {imputed_data_path}")
