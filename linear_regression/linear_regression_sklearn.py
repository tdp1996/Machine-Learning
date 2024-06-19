import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#example 1
data = pd.read_csv('basic_concept/data_for_lr.csv')

# Drop the missing values
data = data.dropna()

# training dataset and labels
X_train= np.array(data.x[0:500]).reshape(500, 1)
y_train = np.array(data.y[0:500]).reshape(500, 1)

# valid dataset and labels
X_test = np.array(data.x[500:700]).reshape(199, 1)
y_test = np.array(data.y[500:700]).reshape(199, 1)

model = LinearRegression()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)
print("slope:", model.coef_[0])
print("intercept:", model.intercept_)
plt.figure(figsize=(14, 10))
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()


