import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans

# Sample data (replace this with your actual data)
data = {
    'Feature1': [1.2, 2.3, 3.0, 4.1, 5.2, 6.5, 7.0, 8.1],
    'Feature2': [0.1, 1.0, 1.2, 2.1, 2.9, 3.5, 4.0, 4.8],
    'Feature3': [10, 20, 30, 40, 50, 60, 70, 80],
    'Feature4': [0.5, 1.5, 2.0, 3.2, 4.0, 4.9, 6.0, 7.2]
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Standardize the data (mean=0, variance=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply PCA to reduce dimensionality to 2 components (you can adjust this)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=2)
dbscan_labels = dbscan.fit_predict(pca_result)

# K-means clustering (assuming 2 clusters, you can adjust this)
kmeans = KMeans(n_clusters=2, n_init=10)  # Setting n_init to a positive integer
kmeans_labels = kmeans.fit_predict(pca_result)

# Visualize the clustering results
plt.figure(figsize=(12, 6))

# Plot DBSCAN clusters
plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')

# Plot K-means clusters
plt.subplot(1, 2, 2)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-means Clustering')

plt.show()
