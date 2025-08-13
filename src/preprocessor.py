import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

def apply_transformations(df_imputed):
    """
    Applies log1p transformation to highly skewed features as identified in EDA.

    Args:
        df_imputed (pandas.DataFrame): The input DataFrame with imputed values.

    Returns:
        pandas.DataFrame: DataFrame with specified features transformed.
    """
    df_transformed = df_imputed.copy()

    # Features identified as highly skewed during EDA
    # Ensure these columns contain only non-negative values before log1p
    # Our imputation with median ensures this for the '0' columns.
    skewed_features = ['Insulin','DiabetesPedigreeFunction', 'Pregnancies','Age','SkinThickness','Glucose','BMI']

    for col in skewed_features:
        if col in df_transformed.columns:
            # np.log1p(x) computes log(1+x), which handles 0 values gracefully
            df_transformed[col] = np.log1p(df_transformed[col])
            print(f"Applied log1p transformation to: {col}")
    
    return df_transformed

def scale_features(df_transformed, scaler_type='RobustScaler'):
    """
    Scales numerical features using either StandardScaler or RobustScaler.

    Args:
        df_transformed (pandas.DataFrame): The input DataFrame with transformed features.
        scaler_type (str): Type of scaler to use ('StandardScaler' or 'RobustScaler').

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: DataFrame with scaled features.
            - sklearn.preprocessing.Scaler: The fitted scaler object.
    """
    df_scaled = df_transformed.copy()
    
    # Exclude the 'Outcome' column from scaling
    features_to_scale = df_scaled.drop(columns=['Outcome']).columns

    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
        print("Using StandardScaler for feature scaling.")
    elif scaler_type == 'RobustScaler':
        scaler = RobustScaler()
        print("Using RobustScaler for feature scaling.")
    else:
        raise ValueError("Invalid scaler_type. Choose 'StandardScaler' or 'RobustScaler'.")

    df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])
    
    return df_scaled, scaler

# Example usage (for testing the script independently and as a part of pipeline using data_loader.py)
if __name__ == "__main__":
    # For independent testing, we'll simulate loading and imputing data 
    # Uncomment the dummy data codeblock for independent testing
    '''
    # For the standalone test, we'll create a dummy DataFrame.
    # Create a dummy DataFrame that resembles imputed data for testing transformations and scaling
    data = {
        'Pregnancies': [6, 1, 8, 1, 0],
        'Glucose': [148.0, 85.0, 183.0, 89.0, 137.0],
        'BloodPressure': [72.0, 66.0, 64.0, 66.0, 40.0],
        'SkinThickness': [35.0, 29.0, 0.0, 23.0, 35.0], # 0s already handled, this is imputed value
        'Insulin': [169.5, 94.0, 169.5, 94.0, 168.0], # 0s already handled, this is imputed value
        'BMI': [33.6, 26.6, 23.3, 28.1, 43.1],
        'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288],
        'Age': [50, 31, 32, 21, 33],
        'Outcome': [1, 0, 1, 0, 1]
    }
    dummy_df = pd.DataFrame(data)
    print("\nDummy DataFrame (simulating imputed data):")
    print(dummy_df.head())
    
    # Apply transformations to skewed features
    df_transformed = apply_transformations(dummy_df)

    # Scale features
    df_scaled, fitted_scaler = scale_features(df_transformed, scaler_type='RobustScaler') # Or 'StandardScaler'

    print("\nFirst 5 rows of the fully preprocessed data:")
    print(df_scaled.head())
    '''
    
    # For testing the script as a part of pipeline using data_loader.py
    from data_loader import load_and_clean_data,impute_missing_values     # Import from data_loader.py
    
    # Define the path to your raw data (assuming running from src directory for testing)
    raw_data_path = '../data/raw/diabetes.csv'

    # 1. Load and initially clean the data (replace 0s with NaN)
    df_cleaned_nans = load_and_clean_data(raw_data_path)

    # 2. Impute missing values (NaNs)
    df_imputed = impute_missing_values(df_cleaned_nans)

    # 3. Apply transformations to skewed features
    df_transformed = apply_transformations(df_imputed)
    preprocessed_data_path = '../data/processed/diabetes_transformedData.csv'
    df_transformed.to_csv(preprocessed_data_path,index=False)

    # 4. Scale features
    df_scaled, fitted_scaler = scale_features(df_transformed, scaler_type='RobustScaler') # Or 'StandardScaler'

    print("\nFirst 5 rows of the fully preprocessed data:")
    print(df_scaled.head())

    # Save the preprocessed data to the processed folder
    processed_data_path = '../data/processed/diabetes_preprocessedData.csv'
    df_scaled.to_csv(processed_data_path, index=False)
    print(f"\nFully preprocessed data saved to: {processed_data_path}")

    # Save the fitted scaler (if needed for new data prediction)
    joblib.dump(fitted_scaler, '../models/scaler.pkl')
    print("\nFitted scaler saved to: ../models/scaler.pkl")