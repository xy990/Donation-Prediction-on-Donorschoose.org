
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline

#read data
df = pd.read_csv("data_final4.csv")

#Separate the datasets into feature values and target variable
X = df.drop(['funding_status'],axis =1)
Y = df['funding_status']


# In[3]:

#Split the datasets into train data and test data 0.75/0.25
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.75)


# In[4]:

#use OOB error to do model selection
n_est = [50, 100, 200, 500]
m_feat = [1, 3, 6, 31]

aucs_oob = {}
aucs_test = {}

for m in m_feat:
    aucs_oob[m] = []
    aucs_test[m] = []
    for n in n_est:
        rf_oob = RandomForestClassifier(n_estimators=n, max_features=m, oob_score=True)
        rf_oob = rf_oob.fit(X_train, Y_train)
        aucs_oob[m].append(roc_auc_score(Y_train, rf_oob.oob_decision_function_[:,1]))
        aucs_test[m].append(roc_auc_score(Y_test, rf_oob.predict_proba(X_test)[:,1]))


# In[ ]:


#We'll plot in this block

x = np.log2(np.array(n_est))
for m in m_feat:
    plt.plot(x, aucs_oob[m], label='max_feat={}'.format(m))
    
plt.title('OOB AUC by Max Feat and N-Estimators')
plt.xlabel('Log2(N-Estimators)')
plt.ylabel('OOB-AUC')
plt.legend(loc=4, ncol=2, prop={'size':10})
plt.savefig('oob_search.png')

