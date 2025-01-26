# house_price_prediction
The House Price Prediction project uses machine learning and by analyzing historical data, it identifies patterns and trends to provide accurate price predictions. This project supports homebuyers, sellers, and real estate agents in making informed decisions about property values.

## Table of Contents
### 1) Introduction
    1.1) Overview of the Project
    1.2) Objective and Scope
  
### 2) Getting Started
    2.1) Prerequisites

### 3) Features : Summary of Key Analysis and Predictions
    3.1) Exploratory Data Analysis(EDA)
    3.2) High Price Analysis and Hypothesis
    3.3) Feature Engineering
    3.4) Predictive Modeling
    3.5) Hyperparameter Tuning
    3.6) Test Data Prediction and Visualisation

### 4) Data
    4.1) Datasets Used
      4.1.1) Description of datasets.
      4.1.2) Source of data
    4.2) Pre-Processing Steps

### 5) Methodologies
    5.1) Analytical Approaches
    5.2) Predictive Modeling Techniques
    5.3) Tools and Framework Used

### 6) Results
    6.1) Insights Driven from Analytics
    6.2) Key Outcomes from Predictive Modeling

### 7) Visualisation and Images/Conclusion for figures
    7.1) Distribution of Sale Price
    7.2) Correlation Heatmap
    7.3) Predicted vs Actual Values for XGBoost
    7.4) Predicted vs Actual Values for LightBG<
    7.5) XGBoost vs LightBGM Predictions

### 8) Code Structure
    8.1) data_loader.py : Helps in loading the dataset.
    8.2) EDA.py : Exploratory data analysis scripts. 
    8.3) preprocessing.py: Data preprocessing and feature engineering.  
    8.4) model_building.py: Model building and evaluation scripts.  
    8.5) hyperparameter_tuning.py : Hyperparameter tuning for Random Forest, XGBoost, and LightGBM.  
    8.6) test_prediction_and_visualization.py : Test predictions and result visualization.

### 9) Requirements
    9.1) Python 3.8+
    9.2) Required Python Libraries
      9.2.1) Pandas
      9.2.2) Numpy
      9.2.3) matplotlib
      9.2.4) seaborn
      9.2.5) scikit-learn
      9.2.6) xgboost
      9.2.7) lightgbm

### 10) Acknowledgements
### 11) Final Conclusions
---------------------------------------------------------------------------------------------------------------------------------------------------

## 1) Introduction
### 1.1) Overview of the Project
#### House prices will be estimated here, considering such variables as location and size, among others, related to real estate properties. This project incorporates deep exploratory data analysis, followed by the application of modern predictive modeling techniques over identified data trends and relations.

### 1.2) Objective and Scope
#### The key objective is to develop a valid predictive model for house prices that can estimate correctly and thus be helpful in the decision-making analysis of real estate agents and buyers.
---------------------------------------------------------------------------------------------------------------------------------------------------

## 2) Getting Started
### 2.1) Prerequisites 
#### 1) Python 3.8 or higher.
#### 2) Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm  
---------------------------------------------------------------------------------------------------------------------------------------------------

## 3) Features : Summary of Key Analysis and Predictions
### 3.1) Exploratory Data Analysis(EDA)
#### 3.1.1) Distribution analysis of features like SalePrice and GriLivArea.
![EDA Sale Price Distribution](https://drive.google.com/uc?export=view&id=1HZTVvtn6Cp6P1fkIbbm6pvtQqdSVWq-E)
    - The data appears to be right skewed, indicating the need for log normalisation.
    - Applying normalization will improve the data's usability for predictive modeling.
    - Then we apply log normalisation and we get the normal distribution of the data then.
![EDA Sale Price Log Normalied Distribution](https://drive.google.com/uc?export=view&id=1tvzBPaAWIDIxh3JKL76l6Loj_ghPGS6-)
    - The data appears to be highly skewed, indicating the need for log normalization.
    - Applying normalization will improve the data's usability for predictive modeling.

#### 3.1.2) Explanation of the Pairplots
![EDA Pair Plot](https://drive.google.com/uc?export=view&id=1bbl4dPwMKlelJm8W8ON6UeY5YnPBXqOZ)
    - The distribution and the relationship between the key variables can be viewed from the following pairs plot: SalePrice, GrLivArea, TotalBsmtSF, OverallQual, and GarageArea. Most of the variables in the diagonal histograms are right-skewed-for example, SalePrice and GrLivArea have a high concentration of lower values with fewer high-value outliers.
    - Scatterplots highlight strong positive correlations, such as the relationship between SalePrice and GrLivArea or between SalePrice and OverallQual-that is, with larger living areas and higher quality ratings, the sale prices are strongly associated with high values. 
    - Patterns such as clustering in OverallQual and the presence of outliers in SalePrice vs. GrLivArea suggest categorical effects and call for further investigation.
    - These represent the insights that may suggest log normalization for reducing skewness, better outlier handling in modeling, and feature engineering of correlated variables.

#### 3.1.3) Correlation of the features and a heatmap
![Correlation Heatmap](https://drive.google.com/uc?export=view&id=1jE9LoOfUQWtIRcs0dyiHgdALzgyTM3pE)
    - The following correlation heat map depicts the relationship between different numeric features of the dataset. Darker red and blue colors correspond to strong positive and negative correlation, respectively.
    - Some key observations include the following: strong positive correlations between SalePrice and SalePrice_log with OverallQual, GrLivArea, TotalBsmtSF, and GarageArea are critical predictors of house prices.
    - That means features like PoolArea, MiscVal, and Id are poorly or negligibly correlated and perhaps have little effect on SalesPrice. One can observe perfect correlations along the diagonal, since the features are perfectly correlated with themselves. 
    - Generally, this heatmap identifies highly correlated features for inclusion in predictive models, flagging potential risks of multicollinearity among features like GarageCars and GarageArea.

#### 3.1.4) Temporal Analysis
![EDA Temporal Analysis](https://drive.google.com/uc?export=view&id=17gvM-EQmlQqYcQIuFdxl0oEIoSYerv2c)
##### 1) Left Plot : Trend of SalePrice by Year Built
    - The trend of SalePrice depending on a year when the property was built. Generally speaking, the prices of housing are growing, for properties built after 1950 in particular; a serious rise began after the year 2000.
    - Variability of older ones, built before 1950, is bigger because of either historical significance or renovation. 
    - The shaded area around the line gives the amount of variation or confidence interval in sale prices, which decrease as the sale price increases for properties in more recent year's builds, reflecting more consistent pricing.
    - This would tend to suggest that the year built may be related to higher sale prices, perhaps that newer homes generally command higher values.

##### 2) Right Plot : SalePrice by Year Sold 
    - The right plot represents the distribution of SalePrice in properties sold across each year in the period from 2006 through 2010. The median sale price is quite constant for these five years, whereas the dispersion and number of outliers differ among the years.
    - Outliers are concentrated a bit higher in 2008. This may have to do with market instability after the global financial crisis of that year. 
    - The overall distribution indicates that even though the sale prices remained at a standstill during this period, external market factors might have influenced certain transactions to result in the observed outliers.
--------------------------------------------------------------------------------------------------------------------------------------------------

### 3.2) High Price Analysis and Hypothesis
#### 3.2.1) Scatterplot : Saleprice vs GriLivArea for high prices
![SalePrice vs. GrLivArea (Log Transformation)](https://drive.google.com/file/d/1P8827uAwEyEhlTYQ1u2cA723eNVa8GhC/view?usp=share_link)
##### 1) SalePrice vs GrivArea Log Transformation
      - This scatterplot shows the relationship of the logarithm of GrLivArea-the above-ground living area in square feet-against SalePrice. Due to the log transformation of GrLivArea, most of the skewness in the data has been removed, and, therefore, this relationship is more linear. 
      - By the positive correlation value, when GrLivArea increases, then SalePrice is also increased by the importance of having a larger living area for the pricing of the house. 
      - This plot has a clearer trend and is less affected by outliers, which means logarithmic scaling makes the relationship more interpretable.

![SalePrice vs GrivArea (Higher Prices Outlined)](https://drive.google.com/file/d/1_MEGcMzRH7Cb9fW1IvyQkwtTbhO8F2LY/view?usp=share_link)
##### 2) SalePrice vs GrivArea(High Prices Outlined)
      - This scatterplot shows the relationship between SalePrice and GrLivArea. Properties that are considered high-priced (the top 1% based on SalePrice) are in red.
      - Overall, high-priced properties generally are those that have a larger GrLivArea, but some properties command premium prices because of superior quality, prime locations, or custom designs. This plot suggests that high-priced properties are actually not statistical outliers, but a valid market trend. 
      - Isolated high-price points for smaller living areas may indeed reflect a niche market or extraordinary property characteristics.

![SalePrice by Neighbourhood](https://drive.google.com/file/d/1dC-kNTlwkmOVBFU-In6YtmoYSNWSvjyV/view?usp=share_link)
##### 3) Saleprice by Neighbourhood(Without Outliers)
      - The boxplot displays the distribution of SalePrice in different neighborhoods, excluding the statistical outliers. The top median prices include NridgHt, NoRidge, and StoneBr neighborhoods, which denote their premium status, while low medians for places such as MeadowV and IDOTRR suggest a less desirable location or perhaps older property conditions. 
      -  This plot highlights the fact that location is a strong determinant of housing prices, and there are certain neighborhoods that, because of their amenities, reputation, or proximity to key infrastructure, consistently command a higher value.

##### Conclusion for the Above Figures
- From both the log-transformed and raw scatterplots, larger living areas increase house prices considerably. While a majority of higher-priced houses contain more living space, unique features and prime location do raise the price of smaller houses.
-  Neighborhoods play a big role in price: high-priced areas tend to be consistently high because of their prestige and demand, whereas less desirable locations have lower medians.
-  Emphasizing highly priced properties shows the market trends related to premium locations, quality, or design and helps to distinguish a valid trend from an anomaly.
--------------------------------------------------------------------------------------------------------------------------------------------------

### 3.3) Feature Engineering
#### Feature engineering is a crucial step in enhancing the predictive power of models by creating new features that capture hidden patterns in the data. Below are the engineered features and their significance:
##### 3.3.1) Log Transformation
      - Applied log transformation to highly skewed features like GrLivArea, LotArea, 1stFlrSF, TotalBsmtSF, and SalePrice.
      - The objective was to normalize skewed distributions to make them suitable for modeling.
  ![saleprice_relationship_GrLivArea_log](https://drive.google.com/file/d/1P8827uAwEyEhlTYQ1u2cA723eNVa8GhC/view?usp=share_link)
  ![saleprice_relationship_LotArea_log](https://drive.google.com/file/d/1j2J8MBe0Wku-gSJlH11RD4mZv37beE4h/view?usp=share_link)





