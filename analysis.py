#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing data set

dataset = pd.read_csv("data.csv")
x = dataset.iloc[:, 2:-2].values 
y = dataset.iloc[:, -1:].values


#splitting the dataset into training set and test set

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=1/3, random_state=0)


#performing linesr regression 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


#predicting values

y_pred = regressor.predict(x_test)


#backward elimination for those attributes that doesn't have musch effect

import statsmodels.formula.api as sm
x = np.append(arr = np.ones((209,1)).astype(int), values = x, axis = 1)


#removed the 5th attribute here

x_opt = x[:,[0, 1, 2, 3, 4, 6]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary() 


#again performing the linear regression on the model ignoring the unnessesary attributes

x_n = x_opt 
y_n = y
from sklearn.cross_validation import train_test_split
x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(x_n , y_n , test_size=1/3, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train_n,y_train_n)
y_pred_n = regressor.predict(x_test_n)