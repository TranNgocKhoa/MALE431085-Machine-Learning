import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score ,confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn import metrics
import pandas

# Encode categorical features as an integer array.
enc = OrdinalEncoder()
# Encode labels with value between 0 and n_classes-1.
lb_enc = LabelEncoder()

# Importing the dataset
rawData = pandas.read_csv('dataset4classfication.csv', header = 0)
# Thay thế các giá trị NaN bằng Mean
rawData.fillna(rawData.mean())
# Slipt trainning data and labels from rawData
tranningData = rawData[rawData.columns[0:13]]

# The label is the last column (column "Label" is the last column in dataset)
labels = rawData['Label']

# Encode trainning data to number
enc.fit(tranningData)
tranningData = enc.transform(tranningData)

# Encode labels to number
lb_enc.fit(labels)
labels = lb_enc.transform(labels)

# Generate data for testing and training: 70% of trained data and 30% of data are used for testing, and set random_state to randomly select data.
X_train, X_test, y_train, y_test = train_test_split(
    tranningData, labels, test_size=0.3)
max_acc = 0
max_k = 1
max_cf_matrix = []
max_recall = 0
max_pres = 0
max_y =[]
pred = []
predicted = []

X_train, X_test, y_train, y_test = train_test_split(
    tranningData, labels, test_size=0.3)

#for i in range(1, len(X_test)+1,2):
for i in range(1, 200+1,2):
    k = i
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cf_matrix = confusion_matrix(y_test,y_pred)
    recallScore = metrics.recall_score(y_test, y_pred, average="macro")
    precisionScore = metrics.precision_score(y_test, y_pred, average="micro")
    accuracyScore = metrics.accuracy_score(y_test, y_pred)
    if accuracyScore > max_acc:
        max_acc= accuracy_score(y_test, y_pred)
        max_k=k
        max_cf_matrix= cf_matrix
        max_recall = recallScore
        max_pres = precisionScore
        max_y = y_test
        pred = y_pred



# Model Accuracy: how often is the classifier correct?
print("Accuracy:", max_acc)
# Compute the precision
# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
# average="micro" : Calculate metrics globally by counting the total true positives, false negatives and false positives.
print("Precision score:", max_pres)
# Compute the recall
# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives
# average="macro": Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
print("Recall score:", max_recall)
# Compute confusion matrix
print("Confusion Matrix: \n", max_cf_matrix)
print("Max k: ", max_k)