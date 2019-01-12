import matplotlib.pyplot as plt  
import numpy as np 
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn import metrics


def printScore(y_test, y_pred):
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Compute the precision
    # The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
    # average="micro" : Calculate metrics globally by counting the total true positives, false negatives and false positives.
    print("Precision score: ", metrics.precision_score(y_test, y_pred, average="micro"))
    # Compute the recall
    # The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives
    # average="macro": Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    print("Recall score: ", metrics.recall_score(y_test, y_pred, average="macro"))
    # Compute confusion matrix
    print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_pred))

# read file
FILE_NAME = 'dataset_for_logistic_regression.csv'
rawData = pd.read_csv(FILE_NAME, header=0)

# Tách dữ liệu
X_train, X_test, Y_train, Y_test = train_test_split(rawData[rawData.columns[0:3]], rawData[rawData.columns[3]], test_size = 0.2, random_state = 0)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
printScore(Y_test, y_pred)
