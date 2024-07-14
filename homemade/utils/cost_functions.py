from .analysis import calculate_mean
from .classess import Array


def calculate_mean_squared_error(y_true: Array, y_predict: Array) -> float:
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    MSE is a measure of the average squared difference between the actual and predicted values.
    It is used to evaluate the performance of regression models.

    Args:
        y_true (Array): Array of true values.
        y_predict (Array): Array of predicted values.

    Returns:
        float: The Mean Squared Error between y_true and y_predict.

    Example:
    >>> y_true = Array([1, 2, 3])
    >>> y_predict = Array([1.1, 2.2, 2.8])
    >>> mean_squared_error(y_true, y_predict)
    0.02333333333333332
    """
    squared_error = (y_true - y_predict) ** 2
    mse = calculate_mean(squared_error)
    return mse


def calculate_binary_cross_entropy(y_true: Array, y_predict: Array) -> float:
    """
    Calculate the Binary Cross-Entropy (BCE) between true and predicted values.

    BCE is a measure of the difference between two probability distributions
    and is commonly used to evaluate the performance of classification models.

    Args:
        y_true (Array): Array of true values (binary labels).
        y_predict (Array): Array of predicted probabilities.

    Returns:
        float: The Binary Cross-Entropy between y_true and y_predict.

    Example:
    >>> y_true = Array([1, 0, 1])
    >>> y_predict = Array([0.9, 0.1, 0.8])
    >>> binary_cross_entropy(y_true, y_predict)
    0.164252033486018
    """
    total_element = y_true.shape[0]
    total_sum = (y_true * y_predict.log() + (1 - y_true) * (1 - y_predict).log()).sum()
    bce = (-1 / total_element) * total_sum

    return bce
