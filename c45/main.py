#!/usr/bin/env python
import pdb
import pandas as pd
import numpy as np
from c45 import C45
from sklearn import metrics
from sklearn.model_selection import train_test_split


# ==========================
c1 = C45("traindata.csv", "name.csv")
c1.fetchData()
c1.preprocessData()
c1.generateTree()
c1.printTree()
X_test = pd.read_csv('x_testdata.csv')
X_test = X_test.values.tolist()
Y_test = pd.read_csv('y_testdata.csv')

y_predict = c1.predict(X_test)

from sklearn.preprocessing import LabelEncoder
lb_enc = LabelEncoder()
y_test = lb_enc.fit_transform(Y_test)
y_predict = lb_enc.fit_transform(y_predict)
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))
print("Precision score: ", metrics.precision_score(y_test, y_predict, average="micro"))
# Compute the recall
# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives
# average="macro": Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
print("Recall score: ", metrics.recall_score(y_test, y_predict, average="macro"))
# Compute confusion matrix
print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_predict))
