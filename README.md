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



