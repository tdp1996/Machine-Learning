import matplotlib.pyplot as plt
import pandas as pd
#example 1
data = pd.read_csv('basic_concept/data_for_lr.csv')

# Drop the missing values
data = data.dropna()

# training dataset and labels
X_train= list(data.x[0:500])
y_train = list(data.y[0:500])

# valid dataset and labels
X_test = list(data.x[500:700])
y_test = list(data.y[500:700])

# Calculate the mean of X and y
mean_X = sum(X_train) / len(X_train)
mean_y = sum(y_train) / len(y_train)

# Calculate slope and intercept 
numerator = sum((X_train[i] - mean_X) * (y_train[i] - mean_y) for i in range(len(X_train)))
denominator = sum((X_train[i] - mean_X) ** 2 for i in range(len(X_train)))
slope = numerator / denominator
intercept = mean_y - slope * mean_X
print("slope:", slope)
print("intercept:", intercept)

# Predict function
def predict(x):
    return slope * x + intercept

# Dự đoán giá trị y cho các giá trị X
y_pred = [predict(xi) for xi in X_test]

#visualize results
plt.figure(figsize=(14, 10))
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()


