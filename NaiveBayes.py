# impoet libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.impute import SimpleImputer

# sử dụng Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

# Object để encode features data
enc = OrdinalEncoder()
# Object để encode label data
lb_enc = LabelEncoder()

# Lấy dữ liệu từ file csv
rawData = pd.read_csv('dataset4classfication.csv', header=0)
# Thay thế các giá trị NaN bằng Mean
rawData.fillna(rawData.mean())
# Lấy dữ liệu từ các cột features
tranningData = rawData[rawData.columns[0:13]]

# Lấy dữ liệu cột label
labels = rawData['Label']

# Encode trainning data to number
enc.fit(tranningData)
tranningData = enc.transform(tranningData)
# Scale dữ liệu thành 0-1
scaler = MinMaxScaler()  
tranningData = scaler.fit_transform(tranningData)

# Encode labels to number
lb_enc.fit(labels)
labels = lb_enc.transform(labels)

# Tạo tập dữ liệu training và tập dữ liệu test
X_train, X_test, y_train, y_test = train_test_split(tranningData, labels, test_size=0.2,random_state=109) 

# Training dữ liệu
model = gnb.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Compute the precision
# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
# average="micro" : Calculate metrics globally by counting the total true positives, false negatives and false positives.
print("Precision score:", metrics.precision_score(y_test, y_pred, average="micro"))
# Compute the recall
# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives
# average="macro": Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
print("Recall score:", metrics.recall_score(y_test, y_pred, average="macro"))
# Compute confusion matrix
print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_pred))