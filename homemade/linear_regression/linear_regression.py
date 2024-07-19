"""
This module contains a LinearRegression class that implements a simple linear regression model.
The class supports data normalization, gradient descent training, and prediction functionalities.

Classes:
    LinearRegression: Implements a linear regression model with methods for training and prediction.
"""

from ..utils.analysis import calculate_mean
from ..utils.array import Array
from ..utils.normalize import normalize


class LinearRegression:
    """
    A class to represent a linear regression model.

    Attributes:
        data (Array): The feature matrix.
        labels (Array): The target values.
        slope (Array): The model's slope parameters.
        intercept (float): The model's intercept parameter.
    """

    def __init__(
        self,
        data: Array,
        labels: Array,
        normalize_data=True,
    ):
        """
        Initializes the LinearRegression model with data, labels, and optionally normalizes the data.

        Args:
            data (Array): The feature matrix.
            labels (Array): The target values.
            normalize_data (bool, optional): Whether to normalize the data. Defaults to True.
        """
        if normalize_data:
            self.data = normalize(data)
        else:
            self.data = data
        self.labels = labels

        # Initialize model parameters
        num_features = self.data.shape[1]
        self.slope = Array([0] * num_features)
        self.intercept = 0

    def train(self, learning_rate, iterations, stopping_threshold=1e-6):
        """
        Trains the linear regression model using gradient descent.

        Args:
            learning_rate (float): The learning rate for gradient descent.
            iterations (int): The maximum number of iterations for gradient descent.
            stopping_threshold (float, optional): The threshold for stopping gradient descent if the cost difference is below this value. Defaults to 1e-6.

        Returns:
            Tuple[Array, float]: The final slope and intercept of the trained model.
        """
        previous_cost = float("inf")
        iteration = 0
        while iteration < iterations:
            predictions = self.hypothesis(self.data, self.slope, self.intercept)
            current_cost = self.cost_function(predictions)
            if abs(previous_cost - current_cost) <= stopping_threshold:
                break
            previous_cost = current_cost
            if iteration % 1000 == 0:
                print(
                    f"Model parameters after {iteration} iterations: Cost: {current_cost}, slope: {self.slope}, intercept: {self.intercept}"
                )
            self.slope, self.intercept = self.gradient_descent(
                predictions, learning_rate
            )
            iteration += 1
        return self.slope, self.intercept

    def gradient_descent(self, predictions, learning_rate):
        """
        Performs one step of gradient descent to update the model parameters.

        Args:
            predictions (Array): The model's predictions based on current parameters.
            learning_rate (float): The learning rate for gradient descent.

        Returns:
            Tuple[Array, float]: The updated slope and intercept of the model.
        """
        num_samples = self.labels.shape[0]
        errors = self.labels - predictions
        # Calculate the gradients
        slope_derivative = (-2 / num_samples) * (self.data.transpose() @ errors)
        intercept_derivative = (-2 / num_samples) * Array.sum(errors)

        self.slope -= learning_rate * slope_derivative
        self.intercept -= learning_rate * intercept_derivative

        return self.slope, self.intercept

    def cost_function(self, predictions):
        """
        Calculates the cost (mean squared error) of the model's predictions.

        Args:
            predictions (Array): The model's predictions.

        Returns:
            float: The cost (mean squared error) of the predictions.
        """
        square_error = (self.labels - predictions) ** 2
        cost = calculate_mean(square_error)
        return cost

    def predict(self, data, slope, intercept):
        """
        Predicts the target values for a given feature matrix using the specified slope and intercept.

        Args:
            data (Array): The feature matrix.
            slope (Array): The slope parameters of the model.
            intercept (float): The intercept parameter of the model.

        Returns:
            Array: The predicted target values.
        """
        data_processed = normalize(data)
        return self.hypothesis(data_processed, slope, intercept)

    @staticmethod
    def hypothesis(data, slope, intercept):
        """
        Computes the hypothesis (predicted values) for a given feature matrix, slope, and intercept.

        Args:
            data (Array): The feature matrix.
            slope (Array): The slope parameters.
            intercept (float): The intercept parameter.

        Returns:
            Array: The predicted values.
        """
        predictions = data @ slope + intercept
        return predictions
