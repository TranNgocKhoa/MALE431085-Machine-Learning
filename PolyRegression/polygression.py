import matplotlib.pyplot as plt  
import numpy as np 
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import linear_model
import sklearn.metrics as metrics

# read file
FILE_NAME = 'dataset_for_poly_regression.csv'
rawData = pd.read_csv(FILE_NAME, header=0)

# Tách dữ liệu
X_train, X_test, Y_train, Y_test = train_test_split(rawData[rawData.columns[0:14]], rawData[rawData.columns[14]], test_size = 0.2, random_state = 0)

# Khai báo các giá trị của degree từ 1 đến 10
degrees = np.arange(1, 11)
best_clf = None
max_r2 = 0.0
best_degree = 1
for deg in degrees:
    poly_model = Pipeline([('poly', PolynomialFeatures(deg)),('linear', linear_model.LinearRegression(fit_intercept=False))])
    poly_model = poly_model.fit(X_train, Y_train)
    score_poly_trained = poly_model.score(X_test, Y_test)
    if max_r2 < score_poly_trained:
        best_clf = poly_model
        best_degree = deg
        max_r2 = score_poly_trained

print(best_clf)
y_pred = best_clf.predict(X_test)
# R^2 Score
print("==========================")
print("Best Degree: %d" % best_degree)
print("R^2 score: %.2f" % best_clf.score(X_test, Y_test))
print("MAE = %.2f" % metrics.mean_absolute_error(Y_test, y_pred))
print("MSE = %.2f" % metrics.mean_squared_error(Y_test, y_pred))
print("RMSE = %.2f" % np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

