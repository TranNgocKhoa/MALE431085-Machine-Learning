import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import svm
from sklearn.svm import SVC
import numpy as np

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


# read file
FILE_NAME = 'dataset_for_PCA_LDA.csv'
rawData = pd.read_csv(FILE_NAME, header=0)

lb_enc = LabelEncoder()
scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
X = rawData[rawData.columns[0:39]]
Y = rawData[rawData.columns[39]]
X = scaler.fit_transform(X)
Y = lb_enc.fit_transform(Y)

np.set_printoptions(precision=39)
mean_vectors = []
for cl in range(0, 2):
    mean_vectors.append(np.mean(X[Y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl]))
S_W = np.zeros((39,39))
for cl,mv in zip(range(0,2), mean_vectors):
    class_sc_mat = np.zeros((39,39))                  # scatter matrix for every class
    for row in X[Y == cl]:
        row, mv = row.reshape(39,1), mv.reshape(39,1) # make column vectors
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat                             # sum class scatter matrices

overall_mean = np.mean(X, axis=0)
S_B = np.zeros((39,39))
for i,mean_vec in enumerate(mean_vectors):  
    n = X[Y==i,:].shape[0]
    mean_vec = mean_vec.reshape(39,1) # make column vector
    overall_mean = overall_mean.reshape(39,1) # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# Make a list of (eigenvalue, eigenvector) tuples
eig_vals = [np.abs(eig_vals[i]) for i in range(len(eig_vals))]
np.savetxt("eigenvals.csv", eig_vals, delimiter=",")
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


# Tách dữ liệu
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
bestClf = findBestClf(X_train, Y_train)
#Predict the response for test dataset
y_pred = bestClf.predict(X_test)
printScore(Y_test, y_pred)

lda = LDA(n_components=2)
X_reduce =  lda.fit_transform(X, Y)
print(X_reduce.shape)
### Code phần 2.2: dùng chính giải thuật A ở trên để làm bài toán classification dựa trên dataset với các LD mới này
X_train, X_test, Y_train, Y_test = train_test_split(X_reduce, Y, test_size=0.2, random_state=0)
bestClf = findBestClf(X_train, Y_train)
#Predict the response for test dataset
y_pred = bestClf.predict(X_test)
printScore(Y_test, y_pred)

#In ra kết quả sau khi Giảm chiều
plt.scatter(X_train[:, 0], X_train[:, 0],c=Y_train)
plt.show()

print("Giải thuật LDA là giải thuật Supervised learning")
print("LDA quan tâm đến X và Y")
print("Cho nên sau khi fit, dữ liệu sẽ được giảm chiều, cùng với đó fit với label")
print("Như vậy kết quả phân loại cũng sẽ tốt hơn.")
