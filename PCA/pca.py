import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

def findBestClf(x_train, y_train):
    degrees = np.arange(0, 11)
    max_accuracy = 0.0
    bestClassifier = None
    for deg in degrees:
        svclassifier = SVC(kernel='poly', gamma=0.5, degree=deg)  
        svclassifier.fit(x_train, y_train)
        #Predict the response for test dataset
        y_pred = svclassifier.predict(X_test)
        accuracy = metrics.accuracy_score(Y_test, y_pred)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            bestClassifier = svclassifier
    return bestClassifier

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


# Đọc file
FILE_NAME = 'dataset_for_PCA_LDA.csv'
rawData = pd.read_csv(FILE_NAME, header=0)

lb_enc = LabelEncoder()
scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
X = rawData[rawData.columns[0:39]]
Y = rawData[rawData.columns[39]]
X = scaler.fit_transform(X)
# lb_enc.fit(Y)
# print(Y)

cov_mat = np.cov(X.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

print('Tỉ lệ đóng góp theo thứ tự giảm dần:')
for value in var_exp:
    print(value)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(10, 6))
    plt.bar(range(39), var_exp, alpha=0.5, align='center',label='individual explained variance')
    plt.step(range(39), cum_var_exp, where='mid',label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()

# Thực hiện Phân loại
# Tách dữ liệu
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# Tìm Clf tốt nhất
bestClf = findBestClf(X_train, Y_train)
#Predict the response for test dataset
y_pred = bestClf.predict(X_test)
printScore(Y_test, y_pred)

# Áp dụng PCA
pca = PCA(n_components=2)
X_reduce = pca.fit_transform(X)
# Tách dữ liệu
X_train, X_test, Y_train, Y_test = train_test_split(X_reduce, Y, test_size = 0.2, random_state = 0)
bestClf = findBestClf(X_train, Y_train)

#Predict the response for test dataset
y_pred = bestClf.predict(X_test)
printScore(Y_test, y_pred)

print("Giải thuật PCA là giải thuật Unsupervised learning")
print("PCA chỉ quan tâm đến X mà không quan tâm Y")
print("Cho nên hầu hết các trường hợp, việc giảm chiều sẽ mất dữ liệu")
print("và ảnh hưởng tiêu cực đến chỉ số đánh giá giải thuật.")


# giá trị interval
h = .01
# Tạo các giá trị để xác định khung sẽ tô màu
x_min, x_max = min(X_test[:, 0]) - 1 ,  max(X_test[:, 0]) + 1
y_min, y_max = min(X_test[:,  1]) - 1 ,  max(X_test[:, 1]) + 1

# Ma trận điểm trong khung tô màu
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

# print(xx.shape)
# print(yy.shape)
# print(clf.predict(np.array([[0.633, 1],[7.1, 1]]).transpose()))
Z = bestClf.predict(np.c_[xx.ravel(), yy.ravel()])
lb_enc = LabelEncoder()
Z = lb_enc.fit_transform(Z)
# Z là kết quả của việc sử dụng model để dự đoán các điểm trong xx và yy
Z = Z.reshape(xx.shape)
# Vẽ đồ thị kiểu contourf, contourf là một kiểu vẽ đồ thị
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

#plt.scatter(X_Train[:, 0],X_Train[:, 1],c=y_train)
plt.scatter(X_test[:, 0],X_test[:, 1],c=lb_enc.fit_transform(Y_test))
plt.show()