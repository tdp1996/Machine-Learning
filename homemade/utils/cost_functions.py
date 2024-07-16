from .analysis import calculate_mean
from .array import Array


class MeanSquaredError:
    """
    Class to calculate the Mean Squared Error (MSE) between true values and predicted values.

    The class can be used as a callable to compute the MSE. The MSE is a measure of the average squared
    difference between the estimated values and the actual value.

    Methods:
        __call__(y_true: Array, y_predict: Array) -> float:
            Calculates and returns the MSE between the true and predicted values.
    """

    def __call__(self, y_true: Array, y_predict: Array) -> float:
        """
        Calculate the Mean Squared Error (MSE) between the true values and predicted values.

        Args:
            y_true (Array): Array of true values.
            y_predict (Array): Array of predicted values.

        Returns:
            float: The Mean Squared Error (MSE) between the true and predicted values.

        Example:
            >>> y_true = Array([3, -0.5, 2, 7])
            >>> y_predict = Array([2.5, 0.0, 2, 8])
            >>> mse = MeanSquaredError()
            >>> mse(y_true, y_predict)
            0.375
        """
        squared_error = (y_true - y_predict) ** 2
        mse = calculate_mean(squared_error)
        return mse


class BinaryCrossEntropy:
    """
    Class to calculate the Binary Cross-Entropy (BCE) loss between true values and predicted values.

    The class can be used as a callable to compute the BCE loss. The BCE loss measures the performance of
    a classification model whose output is a probability value between 0 and 1.

    Methods:
        __call__(y_true: Array, y_predict: Array) -> float:
            Calculates and returns the BCE loss between the true and predicted values.
    """

    def __call__(self, y_true: Array, y_predict: Array) -> float:
        """
        Calculate the Binary Cross-Entropy (BCE) loss between the true values and predicted values.

        Args:
            y_true (Array): Array of true values.
            y_predict (Array): Array of predicted values.

        Returns:
            float: The Binary Cross-Entropy (BCE) loss between the true and predicted values.

        Example:
            >>> y_true = Array([1, 0, 1, 0])
            >>> y_predict = Array([0.9, 0.1, 0.8, 0.2])
            >>> bce = BinaryCrossEntropy()
            >>> bce(y_true, y_predict)
            0.164252033486018
        """
        total_element = y_true.shape[0]
        total_sum = Array.sum(
            y_true * Array.log(y_predict) + (1 - y_true) * Array.log(1 - y_predict)
        )
        bce = (-1 / total_element) * total_sum
        return bce
