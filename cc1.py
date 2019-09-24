# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 09:59:50 2019

@author: Good Guys
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cc = pd.read_csv("C:\\Users\\Good Guys\\Desktop\\pRACTICE\\EXCELR PYTHON\\Assignment\\Linear Regression\\calories_consumed.csv")

#Dividing the data set into x any y
X = cc.iloc[:,:1].values
Y = cc.iloc[:,1].values

#Spliting data into Training and testing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
 X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 7/14, random_state = 0)
#Implementing Classifier
 slm = LinearRegression()
slm.fit(X_train, Y_train)
slm.score(X_train, Y_train)
slm.coef_
Y_predict = slm.predict(X_test)
#Implement the graph
plt.scatter(X_train, Y_train, color = 'blue')
plt.plot(X_train, slm.predict(X_train))
plt.show()
