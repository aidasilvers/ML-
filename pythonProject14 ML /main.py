import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = pd.read_excel('/Users/aidaabilzhanova/Downloads/midterm_ml.xlsx')

# features and target variable
X = data[['Наименование региона', 'Пол']]
y = data['Всего']


X_encoded = pd.get_dummies(X, columns=['Наименование региона', 'Пол'])


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


linear_model = LinearRegression()


linear_model.fit(X_train, y_train)


y_pred = linear_model.predict(X_test)


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values (Linear Regression)")
plt.show()
