# MACHINE LEARNING



This repository contains implementations of various machine learning algorithms and utilities. The aim of this project is to provide a clear and understandable implementation of these algorithms from scratch, focusing on the underlying principles rather than using pre-built libraries.

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
│   ├── linear_regression.py
│
├── logistic_regression/
│   └── logistic_regression.py
│
├── optimization/
│   └── gradient_descent.py
│
├── tests/
│   ├── test_cost_functions.py
│   ├── test_mean_median.py
│   └── test_standard_deviation.py
│
├── utilities/
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
from linear_regression.linear_regression import LinearRegression
import matplotlib.pyplot as plt

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

weight, bias = LinearRegression(X_train=x, y_train=y, learning_rate=0.01)

def myfunc(x):
  return weight[0] * x + bias

mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
```

## Implemented Algorithms
* Linear Regression:
  * `linear_regression.py`
* Logistic Regression:
  *  `logistic_regression.py`
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
