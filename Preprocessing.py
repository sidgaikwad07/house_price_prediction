# Data Pre-Processing file for House Price Prediction
# Author: Siddhant Gaikwad
# Date: 20 January 2025
# Description: This script conatins the pre-processing of the data which is needed for the analysis

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Pre-processing function
def handle_missing_values(data):
    """
    Filling or handling the missing values.
    
    This function handles missing values in a dataset by:
    - Filling numerical columns with their median value. 
    This ensures that extreme values do not overly influence the imputation.
    
    - Filling categorical columns with their most frequent value (mode). 
    This maintains consistency in categorical data.


    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame with potential missing values.

    Returns
    -------
    data : pd.DataFrame
        The DataFrame with missing values handled.

    """
    # handling the missing values
    # Filling the numerical columns with median
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col].fillna(data[col].median(), inplace=True)
    
    # Filling the categorical values with mode
    for col in data.select_dtypes(include=["object", "category"]).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)
    
    return data

def encode_categorical(data):
    """
    Perform one-hot encoding on categorical columns.
    
    This function converts categorical variables into a binary matrix (one-hot encoding),
    dropping the first level to avoid multicollinearity.                                                                  )

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame with categorical variables.

    Returns
    -------
    pd.DataFrame : The Data Frame with categorical variables encoded as numerical columns.

    """
    data = pd.get_dummies(data, drop_first=True)
    return data

def scale_features(data, scaler=None):
    """
    Scale numerical features using StandardScaler.
    
    This function scales numerical features in the dataset to have a 
    mean of 0 and a standard deviation of 1 using StandardScaler from 
    scikit-learn. If an existing scaler is provided, it applies the 
    same scaling otherwise, it fits a new scaler on the numerical features.

    Steps:
    1. Identifies numerical columns in the dataset.
    2. If no scaler is provided, creates a new StandardScaler and fits it to the numerical columns.
    3. If a scaler is provided, uses it to transform the numerical columns.
    4. Updates the original dataset with the scaled numerical values.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame with numerical features to scale.
    scaler : StandardScaler, optional
        An existing StandardScaler instance. If None, a new scaler is created.

    Returns
    -------
    tuple: A tuple containing:
        - pd.DataFrame: The DataFrame with scaled numerical features.
        - StandardScaler: The scaler used for scaling (new or existing).
    """
    numerical_features = data.select_dtypes(include=[np.number])
    
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_features)
    else:
        scaled_data = scaler.transform(numerical_features)
        
    scaled_df = pd.DataFrame(scaled_data, columns=numerical_features.columns, index=data.index)
    data.update(scaled_df)
    
    return data, scaler

if __name__ == "__main__":
    data = pd.read_csv('/Users/sid/Downloads/house_price_prediction/train.csv')
    data = handle_missing_values(data)
    data = encode_categorical(data)
    data = scale_features(data)
    
    
    
    
    
    
    