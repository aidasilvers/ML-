import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load your dataset from an Excel file
data = pd.read_excel('/Users/aidaabilzhanova/Downloads/midterm_ml.xlsx')

print(data.columns)

# Select features and target variable
X = data[['Наименование региона', 'Пол']]
y = data['Всего']

# Encode categorical variables (one-hot encoding)
X_encoded = pd.get_dummies(X, columns=['Наименование региона', 'Пол'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a decision tree regressor
tree_model = DecisionTreeRegressor()

# Train the model on the training data
tree_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = tree_model.predict(X_test)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(tree_model, filled=True, feature_names=X_encoded.columns)
plt.show()