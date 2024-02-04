import pandas as pd
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt


data = pd.read_excel('/Users/aidaabilzhanova/documents/Data_Extract_From_World_Development_Indicators_Тимур.xlsx')


country_names = data['Country Name']


years_data = data.loc[:, '1995 [YR1995]':'2019 [YR2019]']


n_components = 3
model = NMF(n_components=n_components, init='random', random_state=0)
W = model.fit_transform(years_data)
H = model.components_


data['Cluster'] = W.argmax(axis=1)


cluster_counts = data['Cluster'].value_counts()
plt.bar(cluster_counts.index, cluster_counts)
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Cluster Distribution')
plt.show()


data.to_csv('clustered_data.csv', index=False)
