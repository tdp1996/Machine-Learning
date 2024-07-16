from typing import Optional
from .analysis import calculate_mean
from .array import Array


class Metrics:
    """
    A class to calculate various evaluation metrics for machine learning models.

    Attributes:
        y_true (Array): Array of true values.
        y_predict (Array): Array of predicted values.
    """

    def __init__(self, y_true: Array, y_predict: Array):
        """
        Initializes the Metrics object with true and predicted values.

        Args:
            y_true (Array): Array of true values.
            y_predict (Array): Array of predicted values.
        """
        self.y_true = y_true
        self.y_predict = y_predict

    @staticmethod
    def r_square(y_true, y_predict):
        """
        Calculate the R^2 (coefficient of determination) regression score.

        Args:
            y_true (Array): Array of true values.
            y_predict (Array): Array of predicted values.

        Returns:
            float: The R^2 score.
        """
        mean_y_true = calculate_mean(y_true)
        numerator = Array.sum((y_true - y_predict) ** 2)
        denominator = Array.sum((y_true - mean_y_true) ** 2)
        return 1 - (numerator / denominator)

    @staticmethod
    def mean_squared_error(y_true, y_predict):
        """
        Calculate the mean squared error (MSE) between true and predicted values.

        Args:
            y_true (Array): Array of true values.
            y_predict (Array): Array of predicted values.

        Returns:
            float: The mean squared error.
        """
        squared_error = (y_true - y_predict) ** 2
        mse = calculate_mean(squared_error)
        return mse

    @staticmethod
    def mean_absolute_error(y_true, y_predict):
        """
        Calculate the mean absolute error (MAE) between true and predicted values.

        Args:
            y_true (Array): Array of true values.
            y_predict (Array): Array of predicted values.

        Returns:
            float: The mean absolute error.
        """
        absolute_error = Array.abs(y_true - y_predict)
        mae = calculate_mean(absolute_error)
        return mae

    @staticmethod
    def accuracy_score(y_true, y_predict):
        """
        Calculate the accuracy score between true and predicted values.

        Args:
            y_true (Array): Array of true values.
            y_predict (Array): Array of predicted values.

        Returns:
            float: The accuracy score.
        """
        count = 0
        for yt, yp in zip(y_true.data, y_predict.data):
            if yt == yp:
                count += 1
        return count / y_true.shape[0]

    @staticmethod
    def classification_score(
        y_true, y_predict, score_type: str, average: Optional[str] = None
    ):
        """
        Calculate the specified classification score (precision or recall) for the given true labels and predicted labels.

        Args:
            y_true (Array): Array of true values.
            y_predict (Array): Array of predicted values.
            score_type (str): Type of score to calculate. Must be 'precision' or 'recall'.
            average (Optional[str]): Type of averaging performed on the data.
                                     "macro" or "micro" or None (default is None).

        Returns:
            float: Calculated score.

        Raises:
            ValueError: If score_type is not 'precision' or 'recall'.
            ValueError: If average is not one of None, 'macro', or 'micro'.
        """
        if score_type not in ("precision", "recall"):
            raise ValueError("score_type must be either 'precision' or 'recall'")

        if average not in (None, "macro", "micro"):
            raise ValueError("Average must be one of None, 'macro', or 'micro'")

        score = 0
        scores_dict = {}
        classes = set(y_true.data)

        # calculate score for binary classification
        if average is None:
            if score_type == "precision":
                TP = sum(
                    (1 for yt, yp in zip(y_true.data, y_predict.data) if yt == yp == 1)
                )
                FP = sum(
                    (
                        1
                        for yt, yp in zip(y_true.data, y_predict.data)
                        if yt == 0 and yp == 1
                    )
                )
                score = TP / (TP + FP) if (TP + FP) > 0 else 0
            elif score_type == "recall":
                TP = sum(
                    (1 for yt, yp in zip(y_true.data, y_predict.data) if yt == yp == 1)
                )
                FN = sum(
                    (
                        1
                        for yt, yp in zip(y_true.data, y_predict.data)
                        if yt == 1 and yp == 0
                    )
                )
                score = TP / (TP + FN) if (TP + FN) > 0 else 0

        else:  # calculate score for multiclass classification
            for c in classes:
                if score_type == "precision":
                    TP = sum(
                        (
                            1
                            for yt, yp in zip(y_true.data, y_predict.data)
                            if c == yt and c == yp
                        )
                    )
                    FP = sum(
                        (
                            1
                            for yt, yp in zip(y_true.data, y_predict.data)
                            if c != yt and c == yp
                        )
                    )
                    scores_dict[f"precision_{c}"] = (
                        TP / (TP + FP) if (TP + FP) > 0 else 0
                    )
                elif score_type == "recall":
                    TP = sum(
                        (
                            1
                            for yt, yp in zip(y_true.data, y_predict.data)
                            if c == yt and c == yp
                        )
                    )
                    FN = sum(
                        (
                            1
                            for yt, yp in zip(y_true.data, y_predict.data)
                            if c == yt and c != yp
                        )
                    )
                    scores_dict[f"recall_{c}"] = TP / (TP + FN) if (TP + FN) > 0 else 0

            if average == "macro":
                score = sum(scores_dict.values()) / len(classes)
            elif average == "micro":
                TP = sum((1 for yt, yp in zip(y_true.data, y_predict.data) if yt == yp))
                FP_FN = sum(
                    (1 for yt, yp in zip(y_true.data, y_predict.data) if yt != yp)
                )
                score = TP / (TP + FP_FN) if (TP + FP_FN) > 0 else 0

        return score

    @staticmethod
    def recall_score(y_true, y_predict, average=None):
        """
        Calculate the recall score for the given true labels and predicted labels.

        Args:
            y_true (Array): Array of true values.
            y_predict (Array): Array of predicted values.
            average (Optional[str]): Type of averaging performed on the data.
                                     "macro" or "micro" or None (default is None).

        Returns:
            float: The recall score.
        """
        return Metrics.classification_score(y_true, y_predict, "recall", average)

    @staticmethod
    def precision_score(y_true, y_predict, average=None):
        """
        Calculate the precision score for the given true labels and predicted labels.

        Args:
            y_true (Array): Array of true values.
            y_predict (Array): Array of predicted values.
            average (Optional[str]): Type of averaging performed on the data.
                                     "macro" or "micro" or None (default is None).

        Returns:
            float: The precision score.
        """
        return Metrics.classification_score(y_true, y_predict, "precision", average)

    @staticmethod
    def f1_score(y_true, y_predict, average: Optional[str] = None):
        """
        Calculate the f1 score for the given true labels and predicted labels.

        Args:
            y_true (Array): Array of true values.
            y_predict (Array): Array of predicted values.
            average (Optional[str]): Type of averaging performed on the data.
                                     "macro" or "micro" or None (default is None).

        Returns:
            float: The f1 score.
        """
        if average not in (None, "macro", "micro"):
            raise ValueError("Average must be one of None, 'macro', or 'micro'")

        # calculate f1 score for binary classification
        if average is None:
            TP = sum(
                (1 for yt, yp in zip(y_true.data, y_predict.data) if yt == yp == 1)
            )
            FP = sum(
                (
                    1
                    for yt, yp in zip(y_true.data, y_predict.data)
                    if yt == 0 and yp == 1
                )
            )
            FN = sum(
                (
                    1
                    for yt, yp in zip(y_true.data, y_predict.data)
                    if yt == 1 and yp == 0
                )
            )
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * ((precision * recall) / (precision + recall))

        # calculate f1 score for multiclass classification
        else:
            classess = set(y_true.data)
            precision_score = []
            recall_score = []
            sum_TP = 0
            sum_FP = 0
            sum_FN = 0

            for c in classess:
                TP = sum(
                    (
                        1
                        for yt, yp in zip(y_true.data, y_predict.data)
                        if c == yt and c == yp
                    )
                )
                FP = sum(
                    (
                        1
                        for yt, yp in zip(y_true.data, y_predict.data)
                        if c != yt and c == yp
                    )
                )
                FN = sum(
                    (
                        1
                        for yt, yp in zip(y_true.data, y_predict.data)
                        if c == yt and c != yp
                    )
                )
                precision_score.append(TP / (TP + FP) if (TP + FN) > 0 else 0)
                recall_score.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
                sum_TP += TP
                sum_FP += FP
                sum_FN += FN

            if average == "macro":
                f1_score = sum(
                    2 * (p * r) / (p + r) if (p + r) != 0 else 0
                    for p, r in zip(precision_score, recall_score)
                ) / len(classess)

            if average == "micro":
                precision = sum_TP / (sum_TP + sum_FP) if (sum_TP + sum_FP) != 0 else 0
                recall = sum_TP / (sum_TP + sum_FN) if (sum_TP + sum_FN) != 0 else 0
                f1_score = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) != 0
                    else 0
                )

        return f1_score
