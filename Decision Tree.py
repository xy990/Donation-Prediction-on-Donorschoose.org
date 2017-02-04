####### Decision Tree #######
# Author: Han Zhao

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

# plot auc curve 
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
    plt.show()

# read and split data
data = pd.read_csv('data_final4.csv',encoding='utf-8')
X = data.drop(['funding_status'], axis = 1)
Y = data['funding_status']
X_train, X_test ,Y_train, Y_test = train_test_split(X, Y, train_size=.75)

# default, unscaled
tree1 = DecisionTreeClassifier()
tree1.fit(X_train, Y_train1)
plotAUC(Y_test1, tree1.predict_proba(X_test)[:, 1],'DT-1')
# plot feature importance
importances=tree1.feature_importances_
y_pos = np.arange(X_train.shape[1])
plt.figure(figsize=(20,10))
plt.barh(y_pos, importances, align='center', alpha=0.5)
plt.yticks(y_pos, X_train.columns)
plt.xlabel('importance')
plt.title('Feature names')
plt.savefig('Feature_importance1')
plt.show() 

#Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scale = X_train
X_test_scale = X_test
X_train_scale[['students_reached','duration','total_price_including_optional_support']] = scaler.fit_transform(X_train[['students_reached','duration','total_price_including_optional_support']])
X_test_scale[['students_reached','duration','total_price_including_optional_support']] = scaler.fit_transform(X_test[['students_reached','duration','total_price_including_optional_support']])

# default, scaled
tree2 =  DecisionTreeClassifier()
tree2.fit(X_train_scale, Y_train)
plotAUC(Y_test, tree2.predict_proba(X_test_scale)[:, 1],'DT2')

# grid search
param_grid_dt = {'min_samples_leaf':np.linspace(1,80,5,dtype=int), 'min_samples_split':np.linspace(200,1000,5,dtype=int),'max_depth':np.linspace(1,20,5,dtype=int)}
kfolds = KFold(X_train_scale.shape[0], n_folds = 2)
dt_grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv = kfolds)
dt_grid_search.fit(X_train_scale, Y_train)
# fit best on unscaled
dt_best = dt_grid_search.best_estimator_
dt_best.fit(X_train, Y_train)
plotAUC(Y_test, dt_best.predict_proba(X_test)[:, 1],'DT_best')

# learning curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
def plot_learning_curve(estimator, title, X, y, ylim=None, xlim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(1000, 50000, 15,dtype=int)):#(.0001, 0.2, 5)):
    
    plt.figure()
    #tot = 
    #train_sizes = 
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    #plt.set_xlim(
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
title = "Learning Curves (Decision Tree)"
cv = ShuffleSplit(n_splits = 2, test_size=0.2, random_state=0)
estimator = dt_best
plot_learning_curve(estimator, title, X_train, Y_train, ylim=(0.65, 0.75),xlim=(1000,45000), cv=cv, n_jobs=4)
plt.savefig('Learning_Curve_DT.png')
plt.show()

# validation curve
from sklearn.model_selection import validation_curve

param_range = np.linspace(2,1000,5,dtype=int)
train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(), X_train, Y_train, param_name="min_samples_split", param_range=param_range,
    cv=2, scoring="accuracy", n_jobs=4)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with DT")
plt.xlabel("min_samples_split")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
#plt.plot(param_range, train_scores_mean,label="Training score",color = 'g')
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('Validation Curve with DT - min_samples_split.png')
plt.show()

#confusion matrix
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
dt_best_n.fit(X_train_scale, Y_train)
Y_pred = dt_best_n.predict(X_test_scale)
cnf_matrix = confusion_matrix(Y_test, Y_pred)
np.set_printoptions(precision=3)
class_names = [0,1]
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('Confusion matrix-DT_best')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()






