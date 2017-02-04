
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

#Read data
df = pd.read_csv("C:/Users/sherryyang/Desktop/1001term/data_final4.csv")

#Separate the datasets into feature values and target variable
X = df.drop(['funding_status'],axis =1)
Y = df['funding_status']

#Split the datasets into train data and test data 0.75/0.25
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.75)

rfc = RandomForestClassifier()

#Do gridsearch on random forest
param_grid = {'n_estimators': [50,100,200,500]}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid)
CV_rfc.fit(X_train, Y_train)

#Print the parameters in the best model
print(CV_rfc.best_params_)
file = open("gridsearch.txt", "w")
file.write(CV_rfc.best_params_)
file.close()
