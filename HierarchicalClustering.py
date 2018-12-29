import matplotlib.pyplot as plt  
import numpy as np 
import pandas as pd  
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import metrics
import pandas

# Encode categorical features as an integer array.
enc = OrdinalEncoder()

# Importing the dataset
rawData = pandas.read_csv('dataset4classfication.csv', header = 0)
# Thay thế các giá trị NaN bằng Mean
rawData.fillna(rawData.mean())
# Slipt trainning data and labels from rawData
tranningData = rawData

# Encode trainning data to number
enc.fit(tranningData)
tranningData = enc.transform(tranningData)
scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
tranningData = scaler.fit_transform(tranningData)
print("Start clustering")
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', compute_full_tree=True, linkage='ward')  
cluster.fit_predict(tranningData)  


y_hiera = cluster.labels_

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(tranningData)
# Visualising the clusters
plt.scatter(principalComponents[y_hiera == 0, 0], principalComponents[y_hiera == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(principalComponents[y_hiera == 1, 0], principalComponents[y_hiera == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(principalComponents[y_hiera == 2, 0], principalComponents[y_hiera == 2, 1], s = 10, c = 'green', label = 'Cluster 3')
plt.scatter(principalComponents[y_hiera == 3, 0], principalComponents[y_hiera == 3, 1], s = 10, c = 'cyan', label = 'Cluster 4')


plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()