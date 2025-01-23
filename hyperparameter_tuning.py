# Exploratory Data Analysis and Model Building for House Price Prediction
# Author: Siddhant Gaikwad
# Date: 23 January 2025
# Description: This script conatins hyperparameter tuning and evaluation

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def tune_and_evaluate_models(X_train, y_train, X_valid, y_valid):
    """
    Perform hyperparameter tuning for Random Forest, XGBoost, and LightGBM,
    and evaluate their performance on validation data.

    Parameters
    ----------
    X_train : np.array
    Training features.
    
    y_train : pd.Series
    Traning target.
    
    X_valid : np.array
    Validation features.
    
    y_valid : pd.Series
    Validation target.

    Returns
    -------
    Dict : Results containing evaluation metrics for all tuned models.
    """
    
    results = {}
    
    # Random Forest
    try:
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        rf_model = RandomForestRegressor(random_state=42)
        rf_random_search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=rf_param_grid,
            n_iter=50,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        rf_random_search.fit(X_train, y_train)
        best_rf_model = rf_random_search.best_estimator_
        y_pred_rf = best_rf_model.predict(X_valid)
        results['Random Forest'] = {
            'Best Params': rf_random_search.best_params_,
            'RMSE': np.sqrt(mean_squared_error(y_valid, y_pred_rf)),
            'MAE': mean_absolute_error(y_valid, y_pred_rf),
            'R2': r2_score(y_valid, y_pred_rf),
        }
    except Exception as e:
        results['Random Forest'] = {
            'Error': str(e),
        }

    # XGBoost
    try:
        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
        xgb_model = XGBRegressor(random_state=42)
        xgb_random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=xgb_param_grid,
            n_iter=50,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        xgb_random_search.fit(X_train, y_train)
        best_xgb_model = xgb_random_search.best_estimator_
        y_pred_xgb = best_xgb_model.predict(X_valid)
        results['XGBoost'] = {
            'Best Params': xgb_random_search.best_params_,
            'RMSE': np.sqrt(mean_squared_error(y_valid, y_pred_xgb)),
            'MAE': mean_absolute_error(y_valid, y_pred_xgb),
            'R2': r2_score(y_valid, y_pred_xgb),
        }
    except Exception as e:
        results['XGBoost'] = {
            'Error': str(e),
        }

    # LightGBM
    try:
        lgbm_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5],
        }
        lgbm_model = LGBMRegressor(random_state=42)
        lgbm_random_search = RandomizedSearchCV(
            estimator=lgbm_model,
            param_distributions=lgbm_param_grid,
            n_iter=50,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        lgbm_random_search.fit(X_train, y_train)
        best_lgbm_model = lgbm_random_search.best_estimator_
        y_pred_lgbm = best_lgbm_model.predict(X_valid)
        results['LightGBM'] = {
            'Best Params': lgbm_random_search.best_params_,
            'RMSE': np.sqrt(mean_squared_error(y_valid, y_pred_lgbm)),
            'MAE': mean_absolute_error(y_valid, y_pred_lgbm),
            'R2': r2_score(y_valid, y_pred_lgbm),
        }
    except Exception as e:
        results['LightGBM'] = {
            'Error': str(e),
        }

    return results
    
    
    
    
    
    
    
    
    
    