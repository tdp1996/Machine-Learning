# MACHINE LEARNING



This repository contains implementations of various Machine Learning algorithms and utilities. The aim of this project is to provide a clear and understandable implementation of these algorithms from scratch, focusing on the underlying principles rather than using pre-built libraries.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Implemented Algorithms](#implemented-algorithms)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
```plaintext
Machine_Learning_Project/
│
├── linear_regression/
│   ├── data_test
│   └── main.py
│
├── logistic_regression/
│   ├── data_test
│   ├── logistic_regression_binary.py
│   └── logistic_regression_multiclass.py
│
├── metrics/
│   └── metrics.py
│
├── optimization/
│   └── gradient_descent.py
│
├── tests/
│   ├── test_metrics/
│        ├── test_metrics.py
|   ├── test_optimization/
|        ├── test_gradient_descent.py
|   ├── test_utilities/
|        ├── test_activations.py
│        ├── test_cost_functions.py
│        ├── test_mean_median.py
│        └── test_standard_deviation.py
│
├── utilities/
|   ├── activations.py
│   ├── cost_functions.py
│   ├── mean_median_mode.py
│   └── standard_deviation.py
│
├── docs/
│   ├── linear_regression.md
│   ├── logistic_regression.md
│   └── optimization.md
│
├── .gitignore
├── README.md
├── env.yaml
├── requirements.txt
└── setup.py
```

## Installation
To get started with this project, clone the repository and install the necessary dependencies:
```
git clone https://github.com/tdp1996/Machine-Learning.git
conda env create -f env.yaml
conda activate machine_learning
```

## Usage
Each algorithm is implemented in its respective directory. You can run the scripts directly or import the functions and classes in your own scripts.
### Example
```
import matplotlib.pyplot as plt
from linear_regression.main import train_LinearRegression, LinearRegression_model
from metrics.metrics import calculate_r_square

data = pd.read_csv('linear_regression/data_test/simple_linear_regression_data.csv')

# Drop the missing values
data = data.dropna()

# training dataset and labels
X_train= list(data.X[0:80])
y_train = list(data.y[0:80])

# valid dataset and labels
X_test = list(data.X[80:100])
y_test = list(data.y[80:100])

# train model
weight, bias = train_LinearRegression(X_train=X_train, y_train=y_train,iterations=10000, learning_rate= 0.001)

# predict results and evaluate the performance of the model using R square metric
y_predict = LinearRegression_model(X=X_test, weights=weight,bias=bias)
r_square = calculate_r_square(y_true=y_test, y_predict=y_predict)
print(y_predict)
print(r_square)


# visualize results
plt.figure(figsize=(14, 10))
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_predict, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
```

## Implemented Algorithms
* Linear Regression:
  * `main.py`
* Logistic Regression:
  *  `logistic_regression_binary.py`
  *  `logistic_regression_multiclass.py`
* Optimization Algorithms:
  * `gradient_descent.py`

## Testing
The project includes unit tests to ensure the correctness of the implementations. You can run the tests using pytest:
```
pytest tests/
```

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please create an issue or submit a pull request.
### Steps to Contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Make your changes and commit them: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature-branch`.
5. Submit a pull request.
