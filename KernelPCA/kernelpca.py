import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np

def find_best_clf(x_train, y_train):
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


def readData(fileName):
    rawData = pd.read_csv(fileName, header=0)
    lb_enc = LabelEncoder()
    scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    X = rawData[rawData.columns[0:4]]
    X.Gender = X.Gender.replace(['Male', 'Female'], [0, 1]).astype(int)
    Y = rawData[rawData.columns[4]]
    X = scaler.fit_transform(X)
    # Encode labels to number
    lb_enc.fit(Y)
    Y = lb_enc.transform(Y)
    return X, Y


# read file
FILE_NAME = 'dataset_for_KernelPCA.csv'
X, Y = readData(FILE_NAME)

# Tách dữ liệu
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# Sử dụng dữ liệu gốc, áp dụng SVM Poly Kernel
best_clf = find_best_clf(X_train, Y_train)
# Tính toán Confusion matrix
y_pred = best_clf.predict(X_test)
printScore(Y_test, y_pred)

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_reduce = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_reduce)
centered_matrix = X - X.mean(axis=1)[:, np.newaxis]
cov = np.dot(centered_matrix, centered_matrix.T)
eigvals, eigvecs = np.linalg.eig(cov)
np.set_printoptions(threshold=np.inf)
np.savetxt("eigvals.csv", eigvals, delimiter=",")
np.savetxt("eigvecs.csv", eigvecs, delimiter=",")

print("There are %d Eigenvalues" % kpca.lambdas_.shape[0])
# print('100 Eigenvalues in descending order:')
# for (value, i) in zip(kpca.lambdas_, range(100)):
#     print(value)
    

### Code phần 2.2: dùng chính giải thuật A ở trên để làm bài toán classification dựa trên dataset với các LD mới này
X_train, X_test, Y_train, Y_test = train_test_split(X_reduce, Y, test_size=0.2, random_state=0)

best_clf = find_best_clf(X_train, Y_train)

#Predict the response for test dataset
y_pred = best_clf.predict(X_test)
printScore(Y_test, y_pred)