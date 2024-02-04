import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans

# Sample data
data = {
    'сельские': [111, 66, 75, 39, 367, 273, 122, 106, 72, 55, 177, 115, 80, 37, 131, 74, 92, 121, 89, 90, 312, 210, 45, 31, 137, 55, 133, 72,0,0,0,0,0,0],
    '15-17 дети': [3, 6, 7, 14, 10, 25, 18, 15, 6, 7, 7, 11, 3, 12, 9, 7, 10, 14, 12, 7, 22, 8, 0, 6, 3, 3, 12, 8, 9, 9, 6, 10, 11, 11],
    '14-28 вкл.': [36, 33, 55, 82, 102, 113, 98, 93, 52, 53, 63, 59, 38, 40, 58, 60, 59, 91, 70, 58, 97, 66, 29, 34, 30, 23, 57, 64, 79, 86, 83, 89, 66, 61],
    '85+': [1, 0, 0, 1, 2, 1, 0, 0, 0, 1, 1, 1, 3, 3, 0, 3, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 3, 0, 0, 3, 3, 1, 2]
}


df = pd.DataFrame(data)


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)


dbscan = DBSCAN(eps=0.5, min_samples=2)
dbscan_labels = dbscan.fit_predict(pca_result)


kmeans = KMeans(n_clusters=2, n_init=10)  # Setting n_init to a numeric value
kmeans_labels = kmeans.fit_predict(pca_result)


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')


plt.subplot(1, 2, 2)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-means Clustering')

plt.show()
