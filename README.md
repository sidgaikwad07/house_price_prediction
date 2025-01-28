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

### 7) Code Structure
    7.1) data_loader.py : Helps in loading the dataset.
    7.2) EDA.py : Exploratory data analysis scripts. 
    7.3) preprocessing.py: Data preprocessing and feature engineering.  
    7.4) model_building.py: Model building and evaluation scripts.  
    7.5) hyperparameter_tuning.py : Hyperparameter tuning for Random Forest, XGBoost, and LightGBM.  
    7.6) test_prediction_and_visualization.py : Test predictions and result visualization.

### 8) Requirements
    8.1) Python 3.8+
    8.2) Required Python Libraries
      8.2.1) Pandas
      8.2.2) Numpy
      8.2.3) matplotlib
      8.2.4) seaborn
      8.2.5) scikit-learn
      8.2.6) xgboost
      8.2.7) lightgbm

### 9) Acknowledgements
### 10) Final Conclusions
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
![SalePrice vs. GrLivArea (Log Transformation)](https://drive.google.com/uc?export=view&id=1P8827uAwEyEhlTYQ1u2cA723eNVa8GhC)
##### 1) SalePrice vs GrivArea Log Transformation
      - This scatterplot shows the relationship of the logarithm of GrLivArea-the above-ground living area in square feet-against SalePrice. Due to the log transformation of GrLivArea, most of the skewness in the data has been removed, and, therefore, this relationship is more linear. 
      - By the positive correlation value, when GrLivArea increases, then SalePrice is also increased by the importance of having a larger living area for the pricing of the house. 
      - This plot has a clearer trend and is less affected by outliers, which means logarithmic scaling makes the relationship more interpretable.

![SalePrice vs GrivArea (Higher Prices Outlined)](https://drive.google.com/uc?export=view&id=1_MEGcMzRH7Cb9fW1IvyQkwtTbhO8F2LY)
##### 2) SalePrice vs GrivArea(High Prices Outlined)
      - This scatterplot shows the relationship between SalePrice and GrLivArea. Properties that are considered high-priced (the top 1% based on SalePrice) are in red.
      - Overall, high-priced properties generally are those that have a larger GrLivArea, but some properties command premium prices because of superior quality, prime locations, or custom designs. This plot suggests that high-priced properties are actually not statistical outliers, but a valid market trend. 
      - Isolated high-price points for smaller living areas may indeed reflect a niche market or extraordinary property characteristics.

![SalePrice by Neighbourhood](https://drive.google.com/uc?export=view&id=1dC-kNTlwkmOVBFU-In6YtmoYSNWSvjyV)
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
![saleprice_relationship_GrLivArea_log](https://drive.google.com/uc?export=view&id=1P8827uAwEyEhlTYQ1u2cA723eNVa8GhC)
![saleprice_relationship_LotArea_log](https://drive.google.com/uc?export=view&id=1j2J8MBe0Wku-gSJlH11RD4mZv37beE4h)

##### 3.3.2) Interaction Features
        - Created Qual_LivArea by multiplying OverallQual (quality) and GrLivArea (above-ground living area).
        - Created Qual_TotArea by combining OverallQual with the total area (living + basement).
        - The objective was to capture the interaction between quality and area in determining house prices.

##### 3.3.3) Date-Derived Features
        - HouseAge: Difference between the year the house was sold (YrSold) and the year it was built (YearBuilt).
        - RemodelAge: Difference between YrSold and the year of last remodeling (YearRemodAdd).
        - IsRemodeled: Binary feature indicating if the house has been remodeled (1) or not (0).
        - The objective was to assess the impact of house age and remodeling on the prices.

##### 3.3.4) Aggregated Features
        - TotalBathrooms: Sum of all bathrooms, including basement and half bathrooms.
        - TotalPorchSF: Combined area of all porch types.
        - TotalSqFeet: Total square footage including living and basement areas.
        - The objective was to aggregate relevant details to better capture home characteristics.

##### 3.3.5) Neighbourhood-based Features
        - HighPriceNeighborhood: Binary feature identifying neighborhoods with consistently higher prices (NridgHt, NoRidge, StoneBr).
        - The objective was to highlight neighborhoods that command premium prices.

##### 3.3.6) Outlier-based Features
        - IsLuxury: Binary feature indicating houses in the top 1% of prices.
        - The objective was to differentiate luxury houses from regular listings.

##### 3.3.7) Relationship with SalePrice
        - IsLuxury: Luxury houses have significantly higher prices.
        - RemodelAge: Older remodels correlate with lower prices.
        - TotalBathrooms: More bathrooms correlate with higher prices.
![SalePrice_EDA1](https://drive.google.com/uc?export=view&id=1HaPQlrUG7tdNsq8dduuAGTwWbcKyo9Hj)
![SalePrice_EDA2](https://drive.google.com/uc?export=view&id=1KYocqBLI-c-Cy64lNvncaQT5eDnvktNB)
![SalePrice_EDA3](https://drive.google.com/uc?export=view&id=1lKhJZZv12IXUyNTShz1-MzevOtJUqflt)
![SalePrice_EDA4](https://drive.google.com/uc?export=view&id=1XoILBkTEUct6FqdvnIcVXe44Cd7ECVal)
![SalePrice_EDA5](https://drive.google.com/uc?export=view&id=1cDCTuPZSMFenQD0b9tnqtWqMY5ciMW27)
![SalePrice_EDA6](https://drive.google.com/uc?export=view&id=1DxTXlz_jRwipM7tBVXj2HPSKURv0WNdQ)

--------------------------------------------------------------------------------------------------------------------------------------------------

### 3.4) Predictive Modeling
#### i) Predictive modeling is a statistical and machine learning approach used to predict future outcomes or behaviors based on historical data. By identifying patterns and relationships within datasets, predictive models can forecast values or classify observations. 
#### ii) Common applications include customer behavior prediction, risk assessment, healthcare diagnosis, and price forecasting. Predictive modeling stands at the centerpiece of decision-making because it allows business or any other organizations to foresee trends, optimize strategy, and diminish risks. 
#### iii) Among the techniques for providing active insight are regression analysis, decision trees, and ensemble learning; XGBoost, Random Forest-to name a few-have become cornerstones for modern data-driven solutions.
#### iv) In this project, the predictive modeling approach achieved an accuracy of around 78-80% without hyperparameter tuning. This highlights the robustness of the initial models, while also demonstrating the potential for further improvements through hyperparameter optimization, as was evident when fine-tuning significantly enhanced model performance.

- Data Preparation : Prepares the dataset by splitting features and target, handling missing values, and encoding categorical columns.
- Model Training and Evaluation : Builds and evaluates four different models:
    1) Linear Regression
    2) Random Forest
    3) XGBoost
    4) LightGGM
- Feature Importance :  Extracts feature importance for Random Forest.
![Feature Importance](https://drive.google.com/uc?export=view&id=1QzYr1_ZoXluQyivb9fI1YHxY7HCLd6Dp)
--------------------------------------------------------------------------------------------------------------------------------------------------

### 3.5) Hyperparameter Tuning
#### A hyperparameter is a setting that is external to the model, which is set before the training of the model can start, and cannot be learnt from the data. Examples include learning rate, the number of trees in a Random Forest, maximum depth for decision trees, or the number of hidden layers in a neural network. The hyperparameters are very important since they have a great influence on the performance and generalization of the model.
### Importance of Hyperparameter and it's effects on models
#### 1) Model Performance : Properly tuned hyperparameters ensure the model learns the underlying patterns in data effectively, reducing errors and increasing accuracy. Poorly set hyperparameters can lead to underfitting or overfitting.
#### 2) Learning Rate : In gradient-based models (e.g., XGBoost), the learning rate determines how much the model updates in response to errors. A small learning rate results in slower convergence, while a large one may overshoot the optimal solution.
#### 3) Model Complexity : Parameters like maximum tree depth or the number of hidden layers in neural networks define the complexity of the model. Excessive complexity can lead to overfitting, while insufficient complexity can cause underfitting.
#### 4) Optimization & Generalization : Effective hyperparameter tuning, such as adjusting dropout rates in neural networks or setting appropriate subsample ratios in ensemble methods, ensures the model generalizes well to unseen data.

### Observations from Hyperparameter Tuning Results:
Hyperparameter Tuning Results:
Random Forest:
  Best Params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': None}
  RMSE: 29185.1091
  MAE: 17747.0310
  R2: 0.8890
XGBoost:
  Best Params: {'subsample': 0.8, 'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
  RMSE: 25710.3848
  MAE: 16225.5188
  R2: 0.9138
LightGBM:
  Best Params: {'subsample': 0.8, 'reg_lambda': 0.5, 'reg_alpha': 0.1, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.8}
  RMSE: 29660.5220
  MAE: 16969.9308
  R2: 0.8853

- The XGBoost model did the best, providing the lowest RMSE and highest R², whose hyperparameters were such that it best handled the balance between generalization and training accuracy.
- The performance of the Random Forest was quite well, but still lagged somewhat behind XGBoost, since it may fail to capture higher-order patterns so easily.
- LightGBM, while competitive, showed higher RMSE values and thus may still need more tuning or could simply be less efficient on this data compared to XGBoost.
- Thus we choose XGBoost for predicting the prices of houses based on the dataset.
--------------------------------------------------------------------------------------------------------------------------------------------------

### 3.6) Test Predictions and Visualisations
#### Conclusions
#### 1) Model Performace
        - XGBoost and LightGBM have shown very strong predictive powers in the house price prediction task, and besides, the combination of predictions from both models can further enhance the stability and reliability of the results.
        - The Actual vs Predicted plots for both models are well aligned, which indicates low bias and variance between the true values and predicted values.

#### 2) Feature Engineering
        - Feature engineering played a critical role in improving model performance by introducing new features such as Qual_LivArea, TotalBathrooms, and log transformations for skewed variables. These features captured important patterns in the dataset.

#### 3) Test Predictions
        - The test dataset predictions are saved in the test_predictions.csv file, which is available in the GitHub repository. Users can refer to this file to view the predicted sale prices for the test dataset.

#### 4) Future Enhancements
        - Hyperparameter tuning was performed to optimize the models further, but additional fine-tuning and ensembling strategies could be explored for further improvement.
        - Incorporating additional external data (e.g., real estate trends or economic indicators) may also help improve model accuracy.

### Visualisations and Conclusion for the Plots
1) XGBoost vs LightBGM Prediction.
![XgBoost Vs LightBGM](https://drive.google.com/uc?export=view&id=1OqQRRCbMKzOTEM2Ov7ioMie2cunJVrG4)
The scatter plot comparing the predictions from XGBoost and LightGBM indicates a strong linear relationship, with both models producing similar predictions on the test data. The closeness of the points to the red diagonal line suggests that for most data points, the models agree on the predictions, hence making their combined predictions robust and reliable.

2) Actual vs Predicted Values(XGBoost).
![Plot for Actual vs Predicted XGBoost](https://drive.google.com/uc?export=view&id=14c8M_SJW_gKZ8N5HRYHqk-FJfGSchM4e)
The actual vs predicted plot for XGBoost is highly correlated, as the points are concentrated along the red diagonal line. This means that the alignment is very good-that XGBoost makes very good predictions and captures the pattern in the data effectively. A few outliers, which are slightly deviating from the diagonal, show the instances where the model predictions are different from the actual values.

3) Actual vs Predicted Values(LightBGM).
![Plot for Actual vs Predicted LightBGM](https://drive.google.com/uc?export=view&id=14ZmA2V-RAkUA8dk5ErE886GxsoTC-4rS)
The actual vs predicted plot for LightGBM is also strongly correlated, with most points lying almost on the diagonal line. The model overall works fine, though a few outliers can be seen where the predictions are off from the actual values. That indicates LightGBM generalized the data well, though there are minor inaccuracies for specific cases.
--------------------------------------------------------------------------------------------------------------------------------------------------

## 4) Data
#### 4.1) Datasets Used
##### 4.1.1) Description of Datasets
        - train.csv : Contains features and the target variable (`SalePrice`). 
        - test.csv :  Contains features for making predictions. 

##### 4.1.2) Source of Data
        - Kaggle competition dataset: House Prices: Advanced Regression Techniques.

##### 4.2) Pre-Processing Steps
        - Handling missing values using median imputation. 
        - Encoding categorical variables using one-hot encoding.
        - Standardizing numerical features using `StandardScaler`.
--------------------------------------------------------------------------------------------------------------------------------------------------

## 5) Methodologies
### 5.1) Analytical Approaches
        - Exploratory data analysis for understanding feature relationships.  
        - Feature engineering to create meaningful derived features. 

### 5.2) Predictive Modeling Techniques
        - Linear Regression for baseline modeling.  
        - Random Forest, XGBoost, and LightGBM for advanced modeling.  

### 5.3) Tools & Frameworks Used
        - Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm.  
--------------------------------------------------------------------------------------------------------------------------------------------------

## 6) Results
### 6.1) Insights Derived from Analysis
        - Strong correlation observed between `OverallQual` and `SalePrice`.
        - Features like `TotalBathrooms` and `Qual_LivArea` significantly impact house prices.  

### 6.2) Key Outcomes from Predictive Modeling
        - XGBoost with hyperparameter tuning achieved the best performance (R²: 0.9138, RMSE: 25710.38).  
        - Ensemble predictions improved model reliability and accuracy. 
--------------------------------------------------------------------------------------------------------------------------------------------------

## 7) Code Structure
### Giving the code structure 
    7.1) data_loader.py: Script for loading the training and test datasets into Pandas DataFrames.
    7.2) EDA.py: Conducts exploratory data analysis, generates visualizations, and derives initial insights.
    7.3) preprocessing.py: Handles data preprocessing, missing value treatment, feature engineering, and encoding.
    7.4) model_building.py: Implements model building, training, evaluation, and feature importance analysis.
    7.5) hyperparameter_tuning.py: Performs hyperparameter tuning for Random Forest, XGBoost, and LightGBM models.
    7.6) test_prediction_and_visualization.py: Generates predictions on test data and visualizes results, including actual vs predicted plots.
--------------------------------------------------------------------------------------------------------------------------------------------------

## 8) Requirements
### The following libraries are required for the analysis and the prediction of the house dataset.
#### 8.1) Python 3.8+.
#### 8.2) Docker
#### 8.3) Jupyter Notebook
#### 8.4) Python Libraries
        8.4.1) Pandas
        8.4.2) Numpy
        8.4.3) matplotlib
        8.4.4) seaborn
        8.4.5) scikit-learn
        8.4.6) xgboost
        8.4.7) lightgbm
--------------------------------------------------------------------------------------------------------------------------------------------------

## 9) Acknowledgements
    - Special thanks to Kaggle for providing the dataset and to the open-source community for developing the libraries and tools used in this project.
    - Link for Dataset : https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
--------------------------------------------------------------------------------------------------------------------------------------------------

## 10) Conclusion
### This project successfully predicted house prices using EDA, robust preprocessing techniques, feature engineering, and state-of-the-art machine learning models. Advanced machine learning techniques are applied in this project through the combination of Random Forest, XGBoost, and LightGBM. 
### The XGBoost model resulted in the best performance with the highest R² score after hyperparameter tuning on the validation set, at an accuracy of 91.38%. Such comprehensive visualizations included actual vs. predicted plots and model comparisons that provided a deeper view of the model performance. The test predictions themselves and their visualizations reflect the capability of this pipeline to turn out reliable and interpretable results. 
### This project provides an example of how data-driven approaches and machine learning can result in actionable insights for real-world applications. Outputs relevant to the project are made available, including the test_predictions.csv in the GitHub repository.
--------------------------------------------------------------------------------------------------------------------------------------------------
