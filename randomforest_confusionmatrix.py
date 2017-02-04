import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import itertools

#Read data
df = pd.read_csv("data_final4.csv")

#Separate the datasets into feature values and target variable
X = df.drop(['funding_status'],axis =1)
Y = df['funding_status']

#Split the datasets into train data and test data 0.75/0.25
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.75)


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
dt_best = RandomForestClassifier(n_estimators=500)
dt_best.fit(X_train, Y_train)
Y_pred = dt_best.predict(X_test)
cnf_matrix = confusion_matrix(Y_test, Y_pred)
np.set_printoptions(precision=2)
class_names = [0,1]

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('Confusion matrix-DT')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig("cm_randomforest.png")
