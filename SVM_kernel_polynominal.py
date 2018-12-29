import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

# Encode categorical features as an integer array.
enc = OrdinalEncoder()
# Encode labels with value between 0 and n_classes-1.
lb_enc = LabelEncoder()

# Get data from CSV file
rawData = pd.read_csv('dataset4classfication.csv', header = 0)
# Thay thế các giá trị NaN bằng Mean
rawData.fillna(rawData.mean())
# Slipt trainning data and labels from rawData
# Get 1000 rows in the column with an index form 0 to 13
tranningData = rawData.iloc[:, 0:13]

# The label is the last column (column "Label" is the last column in dataset)
labels = rawData['Label']

# Get 1000 rows in the column with an index of 14 (label column)
labels = rawData.iloc[:, 14]

# Encode trainning data to number
enc.fit(tranningData)
tranningData = enc.transform(tranningData)
# Encode labels to number
lb_enc.fit(labels)
labels = lb_enc.transform(labels)
scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
tranningData = scaler.fit_transform(tranningData)
# Generate data for testing and training: 80% of trained data and 20% of data are used for testing, and set random_state to randomly select data.
X_train, X_test, y_train, y_test = train_test_split(tranningData, labels, test_size = 0.20) 


svclassifier = SVC(kernel='poly', gamma=0.5, degree=3)  
svclassifier.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = svclassifier.predict(X_test)


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