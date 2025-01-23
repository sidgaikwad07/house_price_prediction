# Exploratory Data Analysis and Model Building for House Price Prediction
# Author: Siddhant Gaikwad
# Date: 22 January 2025
# Description: This script conatins model building and evaluation

import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from hyperparameter_tuning import tune_and_evaluate_models

def prepare_data(data, target = 'SalePrice', exclude_columns=['Id', 'SalePrice']):
    """
    Prepares data by splitting into features and target, encoding categorical columns,
    and splitting into training and validation sets.

    Parameters
    ----------
    data : pd.DataFrame
    The input dataset.
    
    target_variable : String
    The target column name.
    
    exclude_columns : List
    Columns to exclude from features.

    Returns
    -------
    Tuple: Processed training and validation sets (X_train, X_valid, y_train, y_valid).
    """
    
    X = data.drop(columns=exclude_columns)
    y = data[target]
    
    # Encode the categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    return X_train, X_valid, y_train, y_valid

# Handle some missing values
def handle_missing_values(X_train, X_valid):
    """
    Handles missing values using median imputation
    
    Parameters
    ----------
    X_train : pd.DataFrame
    Training features.
    
    X_valid : pd.DataFrame
    Validation features.

    Returns
    -------
    Tuple: Imputed training and validation features.
    """
    
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_valid_imputed = imputer.transform(X_valid)
    
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
    X_valid_imputed = pd.DataFrame(X_valid_imputed, columns=X_valid.columns, index=X_valid.index)
    return X_train_imputed, X_valid_imputed

# Building & Evaluating Models
def build_and_evaluate_models(X_train, X_valid, y_train, y_valid):
    """
    Builds and evaluates Linear Regression, Random Forest,
    XGBoost, and LightGBM Models.

    Parameters
    ----------
    X_train : np.araray
    Training features (imputed).
    
    X_valid : np.array
    Validation features (imputed).
    
    y_train : pd.Series
    Training target.
    
    y_valid : pd.Series
    Validation target

    Returns
    -------
    dict: Results containing evaluation metrics for all models.
    pd.DataFrame: DataFrame of feature importances (for Random Forest).
    """
    
    results = {}
    
    # Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_valid)
    results['Linear Regression'] = {
        'RMSE': np.sqrt(mean_squared_error(y_valid, y_pred_lr)),
        'MAE': mean_absolute_error(y_valid, y_pred_lr),
        'R2': r2_score(y_valid, y_pred_lr)
    }
    
    # Random Forest model
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_valid)
    results['Random Forest'] = {
        'RMSE': np.sqrt(mean_squared_error(y_valid, y_pred_rf)),
        'MAE': mean_absolute_error(y_valid, y_pred_rf),
        'R2': r2_score(y_valid, y_pred_rf)
    }
    
    # Feature Importance for random forest
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature' : X_train.columns,
        'Importance' : importances}).sort_values(by='Importance', ascending=False)
    
    # XGBoost model
    xgb_model = XGBRegressor(random_state=42, n_estimators=100)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_valid)
    results['XGBoost'] = {
        'RMSE': np.sqrt(mean_squared_error(y_valid, y_pred_xgb)),
        'MAE': mean_absolute_error(y_valid, y_pred_xgb),
        'R2': r2_score(y_valid, y_pred_xgb)
    }
    
    # LightGBM
    lgbm_model = LGBMRegressor(random_state=42, n_estimators=100)
    lgbm_model.fit(X_train, y_train)
    y_pred_lgbm = lgbm_model.predict(X_valid)
    results['LightGBM'] = {
        'RMSE': np.sqrt(mean_squared_error(y_valid, y_pred_lgbm)),
        'MAE': mean_absolute_error(y_valid, y_pred_lgbm),
        'R2': r2_score(y_valid, y_pred_lgbm)}
    
    return results, feature_importance_df

if __name__ == "__main__":
    data = pd.read_csv("/Users/sid/Downloads/house_price_prediction/train.csv")
    X_train, X_valid, y_train, y_valid = prepare_data(data)
    X_train_imputed, X_valid_imputed = handle_missing_values(X_train, X_valid)

    results = tune_and_evaluate_models(X_train_imputed, y_train, X_valid_imputed, y_valid)

    print("\nHyperparameter Tuning Results:")
    for model, metrics in results.items():
        print(f"{model}:")
        if 'Best Params' in metrics:
            print(f"  Best Params: {metrics['Best Params']}")
        if 'RMSE' in metrics:
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  R2: {metrics['R2']:.4f}")
        if 'Error' in metrics:
            print(f"  Error: {metrics['Error']}")
