# Exploratory Data Analysis and Model Building for House Price Prediction
# Author: Siddhant Gaikwad
# Date: 24 January 2025
# Description: This script contains test data predictions and visualization of results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")

def preprocess_test_data(test_data, train_data, scaler, encoded_columns):
    """
    Preprocess the test data to match the structure and scale of training data.

    Parameters
    ----------
    test_data : pd.DataFrame
        Raw test dataset.
    train_data : pd.DataFrame
        Training dataset used for reference.
    scaler : StandardScaler
        Scaler fitted on the training data.
    encoded_columns : Index
        Columns used in the training data after encoding.

    Returns
    -------
    np.array
        Preprocessed and scaled test dataset.
    """
    test_data['LotFrontage'] = test_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())
    test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(0)
    test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(0)

    # Filling missing categorical values with 'None'
    categorical_cols = test_data.select_dtypes(include='object').columns
    for col in categorical_cols:
        test_data[col] = test_data[col].fillna('None')

    # Add engineered features
    test_data['Qual_LivArea'] = test_data['GrLivArea'] * test_data['OverallQual']
    test_data['TotalBathrooms'] = (
        test_data['FullBath'] +
        (0.5 * test_data['HalfBath']) +
        test_data['BsmtFullBath'] +
        (0.5 * test_data['BsmtHalfBath'])
    )
    test_data['TotalPorchSF'] = (
        test_data['OpenPorchSF'] +
        test_data['EnclosedPorch'] +
        test_data['3SsnPorch'] +
        test_data['ScreenPorch']
    )
    test_data['HouseAge'] = test_data['YrSold'] - test_data['YearBuilt']
    test_data['RemodelAge'] = test_data['YrSold'] - test_data['YearRemodAdd']

    # High price neighborhoods
    high_price_neighborhoods = ['NridgHt', 'NoRidge', 'StoneBr']
    test_data['HighPriceNeighborhood'] = test_data['Neighborhood'].apply(
        lambda x: 1 if x in high_price_neighborhoods else 0
    )
    test_data['IsLuxury'] = (test_data['OverallQual'] >= 9).astype(int)

    # Log-transform specific columns
    test_data['GrLivArea_log'] = np.log1p(test_data['GrLivArea'])
    test_data['LotArea_log'] = np.log1p(test_data['LotArea'])

    # Encode categorical variables
    test_data = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)

    # Ensure the test dataset has the same columns as the training dataset
    for col in encoded_columns:
        if col not in test_data.columns:
            test_data[col] = 0  # Add missing columns with 0
    test_data = test_data[encoded_columns]  # Align column order

    # Scale numerical features
    test_data_scaled = scaler.transform(test_data)

    return test_data_scaled

def plot_prediction_distributions(predictions, model_name):
    """Plot the distribution of predicted values."""
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions, kde=True, bins=30, color='blue')
    plt.title(f"Distribution of Predicted Sale Prices ({model_name})")
    plt.xlabel("Predicted Sale Price")
    plt.ylabel("Frequency")
    plt.show()

def compare_models(test_predictions):
    """Compare predictions from XGBoost and LightGBM."""
    plt.figure(figsize=(10, 6))
    plt.scatter(test_predictions['XGBoost_Predictions'], test_predictions['LightGBM_Predictions'], alpha=0.6)
    plt.plot(
        [test_predictions['XGBoost_Predictions'].min(), test_predictions['XGBoost_Predictions'].max()],
        [test_predictions['XGBoost_Predictions'].min(), test_predictions['XGBoost_Predictions'].max()],
        'r--', lw=2
    )
    plt.title("XGBoost vs LightGBM Predictions")
    plt.xlabel("XGBoost Predictions")
    plt.ylabel("LightGBM Predictions")
    plt.show()

if __name__ == "__main__":
    # Load datasets
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    # Prepare training data
    target = 'SalePrice'
    exclude_columns = ['Id', 'SalePrice']
    X = train_data.drop(columns=exclude_columns)
    y = train_data[target]

    # Encode categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    encoded_columns = X_encoded.columns

    # Split training and validation data
    X_train, X_valid, y_train, y_valid = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Scale the training and validation data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    # Train models
    xgb_model = XGBRegressor(random_state=42, n_estimators=300, max_depth=3, learning_rate=0.05)
    xgb_model.fit(X_train, y_train)

    lgbm_model = LGBMRegressor(random_state=42, n_estimators=200, max_depth=3, learning_rate=0.1)
    lgbm_model.fit(X_train, y_train)

    # Preprocess test data
    test_data_processed = preprocess_test_data(test_data, train_data, scaler, encoded_columns)

    # Make predictions
    y_pred_xgb_test = xgb_model.predict(test_data_processed)
    y_pred_lgbm_test = lgbm_model.predict(test_data_processed)

    # Combine predictions
    test_predictions = pd.DataFrame({
        'Id': test_data['Id'],
        'XGBoost_Predictions': y_pred_xgb_test,
        'LightGBM_Predictions': y_pred_lgbm_test
    })
    test_predictions['Final_Predictions'] = (
        test_predictions['XGBoost_Predictions'] + test_predictions['LightGBM_Predictions']
    ) / 2

    # Save predictions
    test_predictions.to_csv("test_predictions.csv", index=False)
    print("Test predictions saved to 'test_predictions.csv'")

    # Visualize distributions
    plot_prediction_distributions(test_predictions['XGBoost_Predictions'], "XGBoost")
    plot_prediction_distributions(test_predictions['LightGBM_Predictions'], "LightGBM")
    compare_models(test_predictions)
    plot_prediction_distributions(test_predictions['Final_Predictions'], "Final Combined Predictions")
