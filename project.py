
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# Importing the Boston House Price Dataset -

house_price_dataset = sklearn.datasets.fetch_california_housing()
print(house_price_dataset, '\n')


# Loading the Dataset to a Pandas DataFrame -
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns= house_price_dataset.feature_names )
print(house_price_dataframe.head(), '\n')


# Add the Target(price) column to the DataFrames -
house_price_dataframe['price'] = house_price_dataset.target
print(house_price_dataframe.head(), '\n')
print(house_price_dataframe.shape, '\n')
print(house_price_dataframe.describe(), '\n')


# checking the missing values -
print(house_price_dataframe.isnull().sum(), '\n')


# Understanding the correlation various features in the Dataset -
correlation = house_price_dataframe.corr()

# Creating a HeatMap to understand the correlation -
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size' :8}, cmap='Blues')
plt.show()


# Splitting the data and target -

X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']
print(X, '\n')
print(Y, '\n')


# Splitting the Data into Training Data and Testing Data -

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(X.shape, X_train.shape, X_test.shape, '\n')
print(Y.shape, Y_train.shape, Y_test.shape, '\n')


# MODEL TRAINING :-

# XGBoost Regressor -
# Logistic Regression -
model = XGBRegressor()

# Training the XGBoostRegressor Model with Training Data -
model.fit(X_train, Y_train)


# MODEL EVALUATION :-

# Accurancy for prediction on Training Data -
Training_data_prediction = model.predict(X_train)

# R Squared error -
score_1 = metrics.r2_score(Y_train, Training_data_prediction)

# Mean Absolute error -
score_2 = metrics.mean_absolute_error(Y_train, Training_data_prediction)

print('R squared error for Training Data :', score_1, '\n')
print('Mean Absolute error for Training Data :', score_2, '\n')


# Accurancy for prediction on Testing Data -
Testing_data_prediction = model.predict(X_test)

# R Squared error -
score_1 = metrics.r2_score(Y_test, Testing_data_prediction)

# Mean Absolute error -
score_2 = metrics.mean_absolute_error(Y_test, Testing_data_prediction)

print('R squared error for Testing Data :', score_1, '\n')
print('Mean Absolute error for Testing Data :', score_2, '\n')


# Visualizing the Actual prices and Predicted prices -

plt.scatter(Y_train, Training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()