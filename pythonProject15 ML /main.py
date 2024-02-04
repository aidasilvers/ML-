import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


data = pd.read_excel('/Users/aidaabilzhanova/documents/Data_Extract_From_World_Development_Indicators_Тимур.xlsx')


selected_columns = [str(year) + ' [YR' + str(year) + ']' for year in range(1995, 2020)]


features = data[selected_columns].apply(pd.to_numeric, errors='coerce')
features = features.dropna()


tsne = TSNE(n_components=2, perplexity=30, random_state=42)



tsne_result = tsne.fit_transform(features)


plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization')
plt.show()
