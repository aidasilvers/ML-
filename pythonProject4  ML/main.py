import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import pandas as pd

# Specify the encoding as 'ISO-8859-1' (latin1)
df = pd.read_csv("/Users/aidaabilzhanova/documents/Diabetes1 .numbers", encoding='ISO-8859-1')


# Step 3: Divide the variables into x and y
x = df.iloc[:, :2]  # Features: BMI and Age
y = df.iloc[:, 2:]  # Target variable: Diabetes

# Step 4: Build a model using sklearn
model = tree.DecisionTreeClassifier(criterion="entropy")
model.fit(x, y)

# Step 5: Model evaluation
score = model.score(x, y)
print("Model Score:", score)

# Step 6: Model Prediction
new_data = [[29, 47]]  # BMI and Age of the person to predict
prediction = model.predict(new_data)
print("Predicted Class:", prediction)

# Step 7: Visualization of the model
plt.figure(figsize=(10, 6))
tree.plot_tree(model, filled=True, feature_names=['BMI', 'Age'], class_names=['No Diabetes', 'Diabetes'])
plt.show()
