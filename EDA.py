# Exploratory Data Analysis and Model Building for House Price Prediction
# Author: Siddhant Gaikwad
# Date: 21 January 2025
# Description: This script conatins EDA of house price dataset, and also outlier analysis

import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt

def eda_plots(data):
    """
    Generate and save EDA plots for analyzing the dataset.

    This function generates the following plots:
    1. Distribution Plot for `SalePrice`:
       - Shows the distribution of the target variable `SalePrice`.
       - Helps understand the skewness of the data.
    
    2. Log-Transformed SalePrice Distribution:
       - Visualizes the log-transformed distribution of `SalePrice`.
       - Useful for making the data closer to a normal distribution, which benefits many ML models.

    3. Correlation Heatmap:
       - Displays the pairwise correlations between all numerical features.
       - Highlights strong correlations with `SalePrice` to identify important predictors.

    4. Temporal Analysis:
       - Analyzes trends of `SalePrice` over time using `YearBuilt` and `YrSold`.
       - Helps explore if older or newer houses have different price distributions.

    5. Pair Plots for Key Features:
       - Generates pairwise scatterplots for selected features (`SalePrice`, `GrLivArea`, `TotalBsmtSF`, `OverallQual`, `GarageArea`).
       - Helps visualize relationships between the target and key predictors.
       

     Args:
        data (pd.DataFrame): The input dataset containing features and the `SalePrice` target.

    Saves:
        - Figures are saved in the "figures/" directory as PNG files:
          - `eda_saleprice_distribution.png`
          - `eda_saleprice_log_distribution.png`
          - `eda_correlation_heatmap.png`
          - `eda_temporal_analysis.png`
          - `eda_pair_plots.png`
          
    Returns
    -------
    None.

    """
    os.makedirs("figures", exist_ok=True)
    
    # Distribution of Sales Price
    plt.figure(figsize=(8,6))
    sns.histplot(data['SalePrice'], kde=True, color='blue', bins=30)
    plt.title('Distribution of SalesPrice')
    plt.xlabel("SalePrice")
    plt.ylabel("Frequency")
    plt.savefig("figures/eda_saleprice_distribution.png")
    plt.close()
    
    # Log-Transformed SalePrice
    data['SalePrice_log'] = np.log(data['SalePrice'])
    plt.figure(figsize=(8,6))
    sns.histplot(data['SalePrice_log'], kde=True, color='green', bins=30)
    plt.title("Log-Transformed SalePrice")
    plt.xlabel("Log(SalePrice)")
    plt.ylabel("Frequency")
    plt.savefig("figures/eda_saleprice_log_distribution.png")
    plt.close()
    
    # Correlation of the features and a heatmap(numerical features)
    plt.figure(figsize=(12, 10))
    numerical_data = data.select_dtypes(include=[np.number])  # Only numeric columns
    corr_matrix = numerical_data.corr()
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, cbar=True)
    plt.title("Correlation Heatmap (Numerical Features Only)")
    plt.savefig("figures/eda_correlation_heatmap.png")
    plt.close()
    
    # Temporal Analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.lineplot(ax=axes[0], data=data, x='YearBuilt', y='SalePrice')
    axes[0].set_title("Trend of SalePrice by Year Built")
    axes[0].set_xlabel("Year Built")
    axes[0].set_ylabel("SalePrice")

    sns.boxplot(ax=axes[1], data=data, x='YrSold', y='SalePrice')
    axes[1].set_title("SalePrice by Year Sold")
    axes[1].set_xlabel("Year Sold")
    axes[1].set_ylabel("SalePrice")

    plt.tight_layout()
    plt.savefig("figures/eda_temporal_analysis.png")
    plt.close()
    
    # Pair Plots for Key Features
    key_features = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'OverallQual', 'GarageArea']
    sns.pairplot(data[key_features])
    plt.savefig("figures/eda_pair_plots.png")
    plt.close()

# Load your data and generate EDA plots
data = pd.read_csv("/Users/sid/Downloads/house_price_prediction/train.csv")
eda_plots(data)

    
    
    
    