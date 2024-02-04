import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
# Load the data
data = pd.read_excel('/Users/aidaabilzhanova/Downloads/ИЭФ.xls.xlsx')

# (columns) to use for t-SNE

features = data[['Exzam1', 'Exzam2', 'Exzam3', 'Exzam4', 'Exzam5', 'Exzam6', 'Exzam7']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(scaled_features)

# Create a new DataFrame with the t-SNE results
tsne_df = pd.DataFrame(data=tsne_result, columns=['Dimension 1', 'Dimension 2'])

# Add the 'Pl-B' and 'Spec' columns from the original data for color-coding
tsne_df['Pl-B'] = data['Pl-B']
tsne_df['Spec'] = data['Spec']

# Scatter plot with color-coding based on 'Pl-B' and 'Spec'
plt.figure(figsize=(10, 8))
plt.scatter(tsne_df['Dimension 1'], tsne_df['Dimension 2'], c=tsne_df['Pl-B'], cmap='viridis', marker='o', s=50)
plt.title('t-SNE Visualization')
plt.colorbar(label='Pl-B')
plt.show()
