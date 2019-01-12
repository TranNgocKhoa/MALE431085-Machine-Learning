import pandas as pd  
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dataset = pd.read_csv("dataset_for_logistic_regression.csv", header=0)  
X = dataset.iloc[:, 0:3].values  
Y = dataset.iloc[:, 3].values 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)  

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)

rfClf = RandomForestClassifier(n_estimators=20, random_state=0,oob_score= True)  
rfClf.fit(X_train, y_train)  
y_pred = rfClf.predict(X_test) 

print("Confusion matrix\n", confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print("Accuracy score:\n", accuracy_score(y_test, y_pred))  