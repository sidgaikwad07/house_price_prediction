# Exploratory Data Analysis and Model Building for House Price Prediction
# Author: Siddhant Gaikwad
# Date: 22 January 2025
# Description: This script conatins Feature engineering for model building

import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def feature_engineering(data):
    """
    Perform feature engineering on the dataset, including log transformation,
    interaction features, date-derived features, and neighborhood-based features.

    Parameters
    ----------
    data : pd.DataFrame
    Input dataset for feature engineering

    Returns
    -------
    pd.DataFrame: Dataset with new engineered features
    """
    
    skewed_features = ['GrLivArea', 'LotArea', '1stFlrSF', 'TotalBsmtSF', 'SalePrice']
    for feature in skewed_features:
        data[feature + '_log'] = np.log1p(data[feature])
        
    # Interaction features
    data['Qual_LivArea'] = data['OverallQual'] * data['GrLivArea']
    data['Qual_TotArea'] = data['OverallQual'] * (data['GrLivArea'] + data['TotalBsmtSF'])
    
    # Date-derived features
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']
    data['RemodelAge'] = data['YrSold'] - data['YearRemodAdd']
    data['IsRemodeled'] = (data['YearBuilt'] != data['YearRemodAdd']).astype(int)

    # Aggregated features
    data['TotalBathrooms'] = (data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath'])
    data['TotalPorchSF'] = (data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch'])
    data['TotalSqFeet'] = data['GrLivArea'] + data['TotalBsmtSF']
    
    # Neighborhood-based features
    prime_neighborhoods = ['NridgHt', 'NoRidge', 'StoneBr']
    data['HighPriceNeighborhood'] = data['Neighborhood'].isin(prime_neighborhoods).astype(int)

    # Outlier-based features
    high_price_threshold = data['SalePrice'].quantile(0.99)
    data['IsLuxury'] = (data['SalePrice'] > high_price_threshold).astype(int)

    print("\nFeature Engineering completed!")
    return data

def validate_features(data, features):
    """
    Validate the distributions and relationships of engineered features.

    Parameters
    ----------
    data : pd.DataFrame
    Dataset with engineered features.
    
    features : List
    List of engineered features to validate.

    Returns
    -------
    - Histogram of feature distributions.
    - Scatterplot of features vs SalePrice.
    - Correlation analysis with SalePrice.
    """
    # Plot distributions of the engineered features
    for feature in features:
        plt.figure(figsize=(10,6))
        sns.histplot(data[feature], kde=True, color='blue')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.savefig(f"figures/feature_distribution_{feature}.png")
        plt.close()
        
    # Correlation with the Sales price
    correlations = data[features + ['SalePrice']].corr()['SalePrice'].sort_values(ascending=False)
    print("\nCorrelation of Engineered Features with SalePrice:")
    print(correlations)
    
    top_corr = correlations[1:7]
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_corr.index, y=top_corr.values, palette='YlGnBu')
    plt.title('Top Engineered features correlated with SalesPrice')
    plt.ylabel("Correlation")
    plt.xlabel("Feature")
    plt.xticks(rotation=45)
    plt.savefig("figures/top_correlations_with_saleprice.png")
    plt.close()

    # Relationship with SalePrice
    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=data[feature], y=data['SalePrice'], alpha=0.6)
        plt.title(f"SalePrice vs {feature}")
        plt.xlabel(feature)
        plt.ylabel("SalePrice")
        plt.savefig(f"figures/saleprice_relationship_{feature}.png")
        plt.close()
        
if __name__ == "__main__":
    data = pd.read_csv("/Users/sid/Downloads/house_price_prediction/train.csv")
    engineered_features = feature_engineering(data.copy())
    validate_features(engineered_features, ['GrLivArea_log', 'LotArea_log', 'Qual_LivArea', 'TotalBathrooms','TotalPorchSF', 'HouseAge', 'RemodelAge', 'HighPriceNeighborhood', 'IsLuxury'])
    
    
    
    
    
    
    
    
    
    