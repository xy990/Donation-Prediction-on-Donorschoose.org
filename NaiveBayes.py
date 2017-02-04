
# coding: utf-8

# In[28]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import nltk
import re
from datetime import datetime
get_ipython().magic('matplotlib inline')
data = pd.read_csv('data_final4.csv')


# In[29]:

X = data.drop(['funding_status'],axis =1)
Y = data['funding_status']


# In[30]:

#Normalize data 
from sklearn import preprocessing
normalized_X = preprocessing.normalize(X)


# In[31]:

#Split data
from sklearn import metrics
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.75)


# In[32]:

from sklearn.naive_bayes import GaussianNB
clf_gb=GaussianNB()
clf_gb.fit(X_train, Y_train)


# In[33]:

#Function for plotting ROC (AUC)
import numpy as np
def plotUnivariateROC(preds, truth, label_string):
    '''
    preds is an nx1 array of predictions
    truth is an nx1 array of truth labels
    label_string is text to go into the plotting label
    '''
    #Student input code here
    #1. call the roc_curve function to get the ROC X and Y values
    fpr, tpr, thresholds = metrics.roc_curve(truth, preds)
    #2. Input fpr and tpr into the auc function to get the AUC
    roc_auc = metrics.auc(fpr, tpr)
    
    #we are doing this as a special case because we are sending unfitted predictions
    #into the function
    if roc_auc < 0.5:
        fpr, tpr, thresholds = roc_curve(truth, -1 * preds)
        roc_auc = auc(fpr, tpr)

    #chooses a random color for plotting
    c = (np.random.rand(), np.random.rand(), np.random.rand())

    #create a plot and set some options
    plt.plot(fpr, tpr, color = c, label = label_string + ' (AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig(label_string+'auc.jpg')
    return roc_auc


# In[35]:

plotUnivariateROC(clf_gb.predict_proba(X_test)[:,1],Y_test, 'GaussianNB Naive Bayes')


# In[ ]:



