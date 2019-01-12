import matplotlib.pyplot as plt  
import numpy as np 
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# read file
FILE_NAME = "dataset_for_multi_linear_regression.csv"
rawData = pd.read_csv(FILE_NAME, header=0)

# Thay thế các ký tự k, M thành số tương ứng.
rawData.Size = (rawData.Size.replace(r"[kM]+$", "", regex=True).astype(float) * 
rawData.Size.str.extract(r"[\d\.]+([kM]+)", expand=False)
.replace(["k","M"], [2**10, 2**20]).astype(int))
#Thay thế các giá trị Install thành số nguyên
rawData.Installs = rawData.Installs.replace(r"[,+]+", "", regex=True).astype(int)

# Tách dữ liệu
X_train, X_test, Y_train, Y_test = train_test_split(rawData[rawData.columns[0:3]], rawData[rawData.columns[3]], test_size = 0.2, random_state = 0)


# Thực hiện trainning sử dụng LinearRegression, có normalize dữ liệu
mlRegression = linear_model.LinearRegression(fit_intercept = True, normalize=True)
mlRegression.fit(X_train, Y_train)

# predict
y_pred = mlRegression.predict(X_test)

# In kết quả
print("Intercept: ", mlRegression.intercept_)
print("Coefficients: ")
print("\ty = %.2f +  %.2f * X1 + %.2f * X2 +  %.2f * X3 \n" % (
    mlRegression.intercept_, mlRegression.coef_[0], mlRegression.coef_[1], mlRegression.coef_[2]))
print("Evaluate model:")
print("MAE = %.2f" % mean_absolute_error(Y_test, y_pred))
print("MSE = %.2f" % mean_squared_error(Y_test, y_pred))
print("RMSE = %.2f" % np.sqrt(mean_squared_error(Y_test, y_pred)))
print("R^2 score: %.2f" % r2_score(Y_test, y_pred))
print("================================================")
