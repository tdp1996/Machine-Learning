from typing import Union
from ..utils.array import Array
from ..utils.normalize import normalize

class LogisticRegression:
    def __init__(
        self,
        data: Array,
        labels: Array,
        normalize_data=True,
    ):
        """
        Initializes the LogisticRegression model with data, labels, and optionally normalizes the data.

        Args:
            data (Array): The feature matrix.
            labels (Array): The target values.
            normalize_data (bool, optional): Whether to normalize the data. Defaults to True.
        """
        if normalize_data:
            precessed_data = normalize(data)
            self.data = precessed_data
        else:
            self.data = data

        self.labels = labels

        # Initialize model parameters
        num_features = self.data.shape[1]
        self.slope = Array([0] * num_features)
        self.intercept = 0

    def train(self, learning_rate, iterations):
        """
        Trains the logistic regression model using gradient descent.

        Args:
            learning_rate (float): The learning rate for gradient descent.
            iterations (int): The maximum number of iterations for gradient descent.

        Returns:
            Union[Tuple[Array, float], List[Tuple[Array, float]]]: The final slope and intercept of the trained model, or a list of parameters for multi-class classification.
        """
        
        classes = list(set(self.labels.data))
        if len(classes) == 2:
            self.slope, self.intercept, current_cost, iteration = self.prepare_for_train(self.labels, learning_rate, iterations)
            print(f"Optimization finished after {iteration} iterations.")
            print(
                f"Final parameters: Cost: {current_cost}, slope: {self.slope}, intercept: {self.intercept}"
            )
            return self.slope, self.intercept
        else:
            model_params = []
            numb_iterations = 0
            for c in classes:
                y_binary_i = Array([1 if y_i == c else 0 for y_i in self.labels.data])
                self.slope, self.intercept, cost_history, current_cost, iteration = self.prepare_for_train(y_binary_i, learning_rate, iterations)
                numb_iterations += iteration
                model_params.append((self.slope, self.intercept))
            print(f"Optimization finished after {numb_iterations} iterations.")
            print(
                f"Final parameters: Cost: {current_cost}, model parameters: {model_params}"
            )
            return model_params    

    def prepare_for_train(self, labels, learning_rate, iterations, stopping_threshold=1e-6):
        """
        Prepares the model for training by performing gradient descent.

        Args:
            labels (Array): The binary target values.
            learning_rate (float): The learning rate for gradient descent.
            iterations (int): The maximum number of iterations for gradient descent.
            stopping_threshold (float, optional): The threshold for stopping gradient descent if the cost difference is below this value. Defaults to 1e-6.

        Returns:
            Tuple[Array, float]: The final slope and intercept of the trained model.
        """
        previous_cost = float('inf')
        iteration = 0

        while iteration < iterations:
            predictions = self.hypothesis(self.data, self.slope, self.intercept)
            current_cost = self.cost_function(labels, predictions)


            if abs(previous_cost - current_cost) <= stopping_threshold:
                break
            previous_cost = current_cost
            if iteration % 1000 == 0:
                print(
                    f"Model parameters after {iteration} iterations: Cost: {current_cost}, slope: {self.slope}, intercept: {self.intercept}"
                )
            self.slope, self.intercept = self.gradient_descent(labels, predictions, learning_rate)
            iteration += 1
        
        return self.slope, self.intercept, current_cost, iteration

    def gradient_descent(self, labels, predictions, learning_rate):
        """
        Performs one step of gradient descent to update the model parameters.

        Args:
            labels (Array): The binary target values.
            predictions (Array): The model's predictions based on current parameters.
            learning_rate (float): The learning rate for gradient descent.

        Returns:
            Tuple[Array, float]: The updated slope and intercept of the model.
        """
        num_samples = labels.shape[0]
        errors = labels - predictions
        # Calculate the gradients
        slope_derivative = (-1 / num_samples) * (self.data.transpose() @ errors)
        intercept_derivative = (-1 / num_samples) * Array.sum(errors)

        self.slope -= learning_rate * slope_derivative
        self.intercept -= learning_rate * intercept_derivative

        return self.slope, self.intercept

    def cost_function(self, labels, predictions):
        """
        Calculates the cost (logistic loss) of the model's predictions.

        Args:
            labels (Array): The binary target values.
            predictions (Array): The model's predictions.

        Returns:
            float: The cost (logistic loss) of the predictions.
        """
        total_elements = labels.shape[0]
        log_predictions = Array.log(predictions)
        log_one_minus_predictions = Array.log(1 - predictions)
        total_sum = Array.sum(labels * log_predictions + (1 - labels) * log_one_minus_predictions)
        return (-1 / total_elements) * total_sum
    
    def sigmoid(z):
        return 1 / (1 + Array.exp(-z))
    
    def predict(self, data, model_params: Union[tuple[Array, float], list[tuple[Array, float]]]) -> Array:
        """
        Predict the class labels for the given data using the trained model.

        Args:
            data (Array): The input features for which to make predictions.
            model_params (Union[Tuple[Array, float], List[Tuple[Array, float]]]): The model parameters
                (slope and intercept) for binary or multiclass classification.

        Returns:
            Array: Predicted class labels.
        """
        processed_data = normalize(data)
        classes = list(set(self.labels.data))

        if len(classes) == 2:
            slope, intercept = model_params
            probabilities = LogisticRegression.hypothesis(processed_data, slope, intercept)
            predictions = Array([1 if prob >= 0.5 else 0 for prob in probabilities.data])
            return predictions
        else:
            probabilities = {}
            for c, (slope, intercept) in enumerate(model_params):
                prob = self.hypothesis(processed_data, slope, intercept)
                probabilities[c] = prob

            # Determine the predicted class for each sample
            predicted_labels = []
            for i in range(processed_data.shape[0]):
                sample_probs = {c: probabilities[c].data[i] for c in probabilities}
                predicted_class = max(sample_probs, key=sample_probs.get)
                predicted_labels.append(predicted_class)

            return Array(predicted_labels)
    
    @staticmethod
    def hypothesis(data, slope, intercept):
        """
        Computes the hypothesis (predicted probabilities) for a given feature matrix, slope, and intercept.

        Args:
            data (Array): The feature matrix.
            slope (Array): The slope parameters.
            intercept (float): The intercept parameter.

        Returns:
            Array: The predicted probabilities.
        """
        linear_combination = data @ slope + intercept
        return LogisticRegression.sigmoid(linear_combination)
