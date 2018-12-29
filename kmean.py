import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

enc = OrdinalEncoder()

# Importing the dataset
dataset = pd.read_csv('dataset4classfication.csv', header=0)
X = dataset
X.fillna(X.mean())
# Encode trainning data to number
enc.fit(X)
X = enc.transform(X)
# Default behavior is to scale to [0,1]
scaler = MinMaxScaler()  
X = scaler.fit_transform(X)



wcss = []
#KMeans()
# init = random, kmean++, kmean++ giúp chọn điểm ban đầu thông minh và giúp việc hội tụ nhanh hơn
# (Xác định tạo số ngẫu nhiên để khởi tạo điểm trung tâm. 
# Sử dụng một số nguyên để làm cho tính ngẫu nhiên xác định)
# max_inter : Maximum number of iterations of the k-means algorithm for a single run (default 300)
# (Số lần lặp tối đa của thuật toán k-means cho một lần chạy, mặc định tối đa là 300)
# precompute_distances : {‘auto’, True, False}, True thì nhanh hơn mà tốn bộ nhớ hơn
# algorithm : “auto”, “full” or “elkan”, default=”auto” Elkan's algorithm


# Đồ thị minh họa số 1
# Using the elbow method to find the optimal number of clusters
    #(Sử dụng phương pháp elbow để tìm số cụm tối ưu)
    # Chọn cụm tối ưu bằng mắt =)))
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42, max_iter= 300, precompute_distances=True)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 4 cụm là tối ưu
kmeans = KMeans(n_clusters = 4, init = 'k-means++',n_init=4, random_state = 42, max_iter= 1000, precompute_distances=True)
kmeans.fit(X)
y_kmeans = kmeans.fit_predict(X)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

#print(transform(X))
print(X)
print(kmeans.fit_predict(X))
# Visualising the clusters
plt.scatter(principalComponents[y_kmeans == 0, 0], principalComponents[y_kmeans == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(principalComponents[y_kmeans == 1, 0], principalComponents[y_kmeans == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(principalComponents[y_kmeans == 2, 0], principalComponents[y_kmeans == 2, 1], s = 10, c = 'green', label = 'Cluster 3')
plt.scatter(principalComponents[y_kmeans == 3, 0], principalComponents[y_kmeans == 3, 1], s = 10, c = 'cyan', label = 'Cluster 4')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()