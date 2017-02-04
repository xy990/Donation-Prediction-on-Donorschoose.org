####### Logistic Regression #######
# Author : Han Zhao

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

def plotAUC(truth, pred, lab):
    fpr, tpr, thresholds = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color=c, label= lab +' (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend(loc=4)
    plt.savefig(lab+'_auc.jpg')

# read data
data = pd.read_csv('data_final4.csv',encoding='utf-8')
Y.value_counts().plot(kind='barh', alpha=0.5)
plt.title('Target variable: funding status')
plt.xlabel('count')
plt.savefig('Funding Status Barplot')
X = data.drop(['funding_status'], axis = 1)
Y = data['funding_status']
X_train, X_test ,Y_train, Y_test = train_test_split(X, Y, train_size=.75)

# logreg with default settings
lr1 = LogisticRegression()
lr1.fit(X_train,Y_train)
plotAUC(Y_test, lr1.predict_proba(X_test)[:, 1],'LR1')
lr1.coef_

# Grid search w/o scaling
kfolds = KFold(X_train.shape[0], n_folds = 5)
param_grid_lr = {'C':[10**i for i in range(-3, 3)], 'penalty':['l1', 'l2']}
lr_grid_search = GridSearchCV(LogisticRegression(), param_grid_lr, cv = kfolds, scoring = 'log_loss')
lr_best1 = lr_grid_search.best_estimator_
lr_best1.fit(X_train,Y_train)
plotAUC(Y_test, lr_best1.predict_proba(X_test)[:, 1],'LR2')

# Grid search w/ scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[['students_reached','duration','total_price_including_optional_support']] = scaler.fit_transform(X_train[['students_reached','duration','total_price_including_optional_support']])
X_test[['students_reached','duration','total_price_including_optional_support']] = scaler.fit_transform(X_test[['students_reached','duration','total_price_including_optional_support']])
kfolds = KFold(X_train.shape[0], n_folds = 5)
param_grid_lr_scale = {'C':[10**i for i in range(-3, 3)], 'penalty':['l1', 'l2']}
lr_grid_search_scaler = GridSearchCV(LogisticRegression(), param_grid_lr_scale, cv = kfolds, scoring = 'log_loss')
lr_best2 = lr_grid_search_scaler.best_estimator_
lr_best2.fit(X_train, Y_train)
plotAUC(Y_test, lr_best2.predict_proba(X_test)[:, 1],'LR3')

# Learning Curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(estimator, title, X, y, ylim=None, xlim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(10000, 15000, 10,dtype=int)):#(.0001, 0.2, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, train_sizes=train_sizes, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

title = "Learning Curves (Logistic Regression)"
cv = ShuffleSplit(n_splits = 10, test_size=0.2, random_state=0)
estimator = LogisticRegression()
#=np.linspace(.0001, 0.2, 5)
plot_learning_curve(estimator, title, X_train, Y_train, ylim=(0.5, 1),xlim=(0,40000), cv=cv, n_jobs=4)
plt.savefig('Learning_Curve_logistic.png')
plt.show()

# confusion Matrix
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
Y_pred = lr1.predict(X_test)
cnf_matrix = confusion_matrix(Y_test, Y_pred)
np.set_printoptions(precision=2)
class_names = [0,1]
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()








