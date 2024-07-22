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
├── homemade
│   ├── data
│   │
│   ├── linear_regression/
│   │   └── linear_regression.py
│   │
│   ├── logistic_regression/
│   │   └── logistic_regression.py
│   │
│   ├── utils/
│   │    ├── analysis.py
│   │    ├── array.py
│   │    ├── metrics.py
│   │    └── normalize.py
│
├── notebook
│   ├── linear_regression/
│   │   ├── multivariate_linear_regression_demo.ipynb   
│   │   └── univariate_linear_regression_demo.ipynb
│   │
│   ├── logistic_regression/
│   │   ├── logistic_regression_binary_demo.ipynb  
│   │   └── logistic_regression_multiclass_demo.ipynb
│
├── tests/
│   ├── test_homemade/
│   │   ├── test_utils/
│   │   │   ├── test_analysis.py
│   │   │   ├── test_classess.py
│   │   │   ├── test_metrics.py
│   │   │   └── test_normalize.py
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
import pandas as pd
from homemade.linear_regression.linear_regression import LinearRegression
from homemade.utils.array import Array
from homemade.utils.metrics import Metrics

data = pd.read_csv('homemade/data/simple_linear_regression_data.csv')

# Drop the missing values
data = data.dropna()

# training dataset and labels
X_train= Array(list(data.X[0:80]))
X_train = Array([[item] for item in X_train.data])
y_train = Array(list(data.y[0:80]))

# valid dataset and labels
X_test = Array(list(data.X[80:100]))
X_test_reshape = Array([[item] for item in X_test.data])
y_test = Array(list(data.y[80:100]))

model = LinearRegression(X_train, y_train)
# train model
weight, bias = model.train(learning_rate= 0.001, iterations=10000)

# predict results and evaluate the performance of the model using R square metric
y_predict = model.predict(X_test_reshape, weight, bias)
r_square = Metrics.r_square(y_true=y_test, y_predict=y_predict)



# visualize results
plt.figure(figsize=(14, 10))
plt.scatter(X_test.data, y_test.data, color='blue', label='Actual data')
plt.plot(X_test.data, y_predict.data, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
```

## Implemented Algorithms
* Linear Regression:
  * `linear_regression.py`
* Logistic Regression:
  *  `logistic_regression.py`

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
