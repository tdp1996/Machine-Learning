from typing import Union
from optimization.gradient_descent import gradient_descent
from utilities.activations import softmax
from utilities.cost_functions import binary_cross_entrophy

STOPPING_THRESHOLD = 1e-6

def train_LogisticRegression(X_train: Union[list[float], list[list[float]]], y_train: list[int], learning_rate: float, stopping_threshold: float = STOPPING_THRESHOLD) -> list[tuple[list[float], float]]:
    pass