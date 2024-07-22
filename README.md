# MACHINE LEARNING


This repository contains implementations of Linear Regression and Logistic Regression algorithms. The main goal is to practice Python programming and understand the fundamentals of these algorithms by building them from scratch.

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
├── homemade/
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
├── notebooks/
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

# Load the data.
data = pd.read_csv("../../data/world-happiness-report-2017.csv")

# Split data set on training and test sets with proportions 80/20.
# Function sample() returns a random sample of items.
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Decide what fields we want to process.
input_param_name = "Economy..GDP.per.Capita."
output_param_name = "Happiness.Score"

# Split training set input and output.
x_train = train_data[[input_param_name]].values
y_train = train_data[output_param_name].values

# Split test set input and output.
x_test = test_data[[input_param_name]].values
y_test = test_data[output_param_name].values

# setup data, convert to Array for training
X_train = Array(x_train.tolist())
Y_train = Array(y_train.tolist())
X_test = Array(x_test.tolist())
Y_test = Array(y_test.tolist())

## Init linear regression instance.
linear_regression = LinearRegression(X_train, Y_train)

learning_rate = 0.01
iterations = 500

slope, intercept, cost_history = linear_regression.train(
    learning_rate=learning_rate, iterations=iterations
)

# test trained model with test data
y_predict = linear_regression.predict(X_test, slope, intercept)

# evaluate the model accuracy with RMSE
mse = Metrics.mean_absolute_error(Y_test, y_predict)
rmse = math.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# prepare to plot result
x_test = x_test.reshape(31,)
y_predict = np.array([[item] for item in y_predict.data])

# visualize results
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, label="Training Dataset")
plt.scatter(x_test, y_test, label="Test Dataset")
plt.plot(x_test, y_predict, color="red", linewidth=2, label="Regression line")
plt.xlabel("Economy..GDP.per.Capita.")
plt.ylabel("Happiness.Score")
plt.title("Countries Happines")
plt.legend()
plt.show()
```

## Implemented Algorithms
* Linear Regression:
  * `linear_regression.py`
* Logistic Regression:
  *  `logistic_regression.py`

## Notebooks
The project includes Jupyter notebooks for demonstrating the algorithms:

* Linear Regression:
  * `multivariate_linear_regression_demo.ipynb`
  * `univariate_linear_regression_demo.ipynb`
* Logistic Regression:
  * `logistic_regression_binary_demo.ipynb`
  * `logistic_regression_multiclass_demo.ipynb`

These notebooks provide step-by-step examples and visualizations to help understand how the algorithms work. You can run these notebooks locally or in an online environment like Google Colab.

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
