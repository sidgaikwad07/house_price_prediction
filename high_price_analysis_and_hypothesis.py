# Exploratory Data Analysis and Model Building for House Price Prediction
# Author: Siddhant Gaikwad
# Date: 22 January 2025
# Description: This script conatins Hypothesis Documentation: The hypothesis is included as part of the function docstring for easy reference.

import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt

def analyze_high_price_houses(data):
    """
    Analyze high-priced houses in the dataset.
    
    Hypothesis:
    High-priced houses are not statistical outliers but legitimate market representations,
    often associated with prime neighborhoods like NridgHt, NoRidge, and StoneBr.
    These properties exhibit features such as larger living areas (GrLivArea) and better
    overall quality (OverallQual), aligning with premium home characteristics. Isolated
    cases, such as a high-priced property in OldTown, may be due to unique factors like
    historical value or custom-built designs.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing housing data
    
    Returns
    -------
        - Scatterplot of Salesprice vs GrLivArea (highlighting high-price properties).
        - Boxplot of SalePrice by Neighborhood.
    """
    
    # Define high price threshold (top 1% of prices)
    high_price_threshold = data['SalePrice'].quantile(0.99)
    high_price_data = data[data['SalePrice'] > high_price_threshold]
    
    #Scatterplot : SalePrice vs GriLivArea for high prices
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=data, x='GrLivArea', y='SalePrice', alpha=0.5, label='All Data')
    sns.scatterplot(data=high_price_data, x='GrLivArea', y='SalePrice', color='red', label='High Price Properties')
    plt.title("SalePrice vs GrLivArea (High Prices Highlighted)")
    plt.xlabel("GrLivArea")
    plt.ylabel("SalePrice")
    plt.legend()
    plt.savefig('figures/high_price_scatterplot.png')
    plt.close()
    
    # Boxplot : SalePrice vs Neighbourhood
    plt.figure(figsize=(12,6))
    sns.boxplot(data=data, x='Neighborhood', y='SalePrice', showfliers=False)
    plt.title("SalePrice by Neighborhood (Without Outliers)")
    plt.xticks(rotation=45)
    plt.savefig("figures/high_price_boxplot_neighborhood.png")
    plt.close()
    
    high_price_neighborhoods = high_price_data['Neighborhood'].value_counts()
    print("\nNeighborhoods with High-Priced Houses:")
    print(high_price_neighborhoods)

    # Summary statistics for high-priced houses
    print("\nSummary Statistics for High-Priced Houses:")
    print(high_price_data[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']].describe())

if __name__ == "__main__":
    data = pd.read_csv("/Users/sid/Downloads/house_price_prediction/train.csv")
    analyze_high_price_houses(data)