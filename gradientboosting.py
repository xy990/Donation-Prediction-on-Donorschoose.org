
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import warnings
import matplotlib.pyplot as plt
%matplotlib inline


#Read Data
df = pd.read_csv("C:/Users/sherryyang/Desktop/1001term/data_final4.csv")


# In[3]:
#Separate the datasets into feature values and target variable
X = df.drop(['funding_status'],axis =1)
Y = df['funding_status']


# In[4]:

#Split the datasets into train data and test data 0.75/0.25
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.75)


# In[1]:
#Fit the gradient boosting with default parameters
GBC = GradientBoostingClassifier()
GBC.fit(X_train, Y_train)

#Plot AUC
def plotAUC(truth, pred, lab):
    fpr, tpr, thresholds = roc_curve(truth, pred, pos_label =1)
    roc_auc = auc(fpr, tpr)
    c1 = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color=c1, label= lab+' (AUC = %0.3f)' % roc_auc)
    #plt.plot(fpr, tpr, color=c1, label= lab+ roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    
plotAUC(Y_test, GBC.predict_proba(X_test)[:,1], 'RFC')

