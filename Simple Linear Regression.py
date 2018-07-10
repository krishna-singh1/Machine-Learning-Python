# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:47:51 2018

@author: KrRish
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df=pd.read_csv("FuelConsumptionCo2.csv")
df.head()

#Lets first have a descriptive exploration on our data.
df.describe()

df.columns

cdf=df[[ 'ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#ploting these features

cdf.hist()
plt.show()

#Now, lets plot each of these features vs the Emission, to see how linear is their relation:
plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color='b')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('Emission')
plt.show()

plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color='r')
plt.xlabel('CYLINDERS')
plt.ylabel('Emission')
plt.show()

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='c')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()

#creating a trainig data set
msk=np.random.rand(len(df)) < 0.8
len(msk)
sum(msk)

train = cdf[msk]
len(train)
test=cdf[~msk]
len(test)

#Creating a simple regression model
'''Linear Regression fits a linear model with coefficients B = (B1, ..., Bn) to 
minimize the 'residual sum of squares' between the independent x in the dataset, 
and the dependent y by the linear approximation. '''

#train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='b')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()

#Modeling Using sklearn package to model data
from sklearn import linear_model
regr=linear_model.LinearRegression()
train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
print('coefficents: ',regr.coef_)
print('Intercept: ' ,regr.intercept_)


#we can plot the fit line over the data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='b')
plt.plot(train_x,regr.coef_[0][0]*train_x+regr.intercept_[0], '-r')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()


'''
we compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.

There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set: 
    - Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
    
    - Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean absolute error because the focus is geared more towards large errors. This is due to the squared term
    exponentially increasing larger errors in comparison to smaller ones.
   
    - Root Mean Squared Error (RMSE).
    - R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. The higher the R-squared, the better the model fits your data. Best possible 
    score is 1.0 and it can be negative (because the model can be arbitrarily worse).

'''

# time to test the model

from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])
test_y_=regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )