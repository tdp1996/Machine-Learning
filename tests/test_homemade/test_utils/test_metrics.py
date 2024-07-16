from homemade.utils.metrics import Metrics
from homemade.utils.array import Array
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
)


def test_r_square():
    y_true = [3, -0.5, 2, 7, 4.2, 1.5, 3.3, 2.8, 1.2, 4.6]
    y_predict = [2.5, 0.0, 2, 8, 4.0, 1.7, 3.5, 2.6, 1.4, 4.8]
    output = Metrics.r_square(y_true=Array(y_true), y_predict=Array(y_predict))
    expected_output = r2_score(y_true, y_predict)
    assert output == expected_output


def test_mean_squared_error():
    y_true = [3, -0.5, 2, 7, 4.2, 1.5, 3.3, 2.8, 1.2, 4.6]
    y_predict = [2.5, 0.0, 2, 8, 4.0, 1.7, 3.5, 2.6, 1.4, 4.8]
    mse = Metrics.mean_squared_error(y_true=Array(y_true), y_predict=Array(y_predict))
    expected_output = mean_squared_error(y_true, y_predict)
    assert mse == expected_output


def test_mean_absolute_error():
    y_true = [3, -0.5, 2, 7, 4.2, 1.5, 3.3, 2.8, 1.2, 4.6]
    y_predict = [2.5, 0.0, 2, 8, 4.0, 1.7, 3.5, 2.6, 1.4, 4.8]
    mae = Metrics.mean_absolute_error(y_true=Array(y_true), y_predict=Array(y_predict))
    expected_output = mean_absolute_error(y_true, y_predict)
    assert mae == expected_output


def test_accuracy_score():
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_predict = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
    score = Metrics.accuracy_score(y_true=Array(y_true), y_predict=Array(y_predict))
    expected_score = accuracy_score(y_true, y_predict)
    assert score == expected_score


def test_precision_score():
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    y_predict = [0, 0, 1, 1, 1, 2, 0, 1, 2, 1]

    precision_score_macro = Metrics.precision_score(
        y_true=Array(y_true), y_predict=Array(y_predict), average="macro"
    )
    expected_precision_score_macro = precision_score(y_true, y_predict, average="macro")
    assert precision_score_macro, expected_precision_score_macro

    precision_score_micro = Metrics.precision_score(
        y_true=Array(y_true), y_predict=Array(y_predict), average="micro"
    )
    expected_precision_score_micro = precision_score(y_true, y_predict, average="micro")
    assert precision_score_micro, expected_precision_score_micro

    y_true_binary = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_predict_binary = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
    precision_score_binary = Metrics.precision_score(
        y_true=Array(y_true_binary), y_predict=Array(y_predict_binary)
    )
    expected_precision_score_binary = precision_score(y_true_binary, y_predict_binary)
    assert precision_score_binary, expected_precision_score_binary


def test_recall_score():
    y_true_binary = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_predict_binary = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]

    recall_score_binary = Metrics.recall_score(
        y_true=Array(y_true_binary), y_predict=Array(y_predict_binary)
    )
    expected_recall_score_binary = recall_score(y_true_binary, y_predict_binary)
    assert recall_score_binary, expected_recall_score_binary

    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    y_predict = [0, 0, 1, 1, 1, 2, 0, 1, 2, 1]
    recall_score_macro = Metrics.recall_score(
        y_true=Array(y_true), y_predict=Array(y_predict), average="macro"
    )
    expected_recall_score_macro = recall_score(y_true, y_predict, average="macro")
    assert recall_score_macro, expected_recall_score_macro

    recall_score_micro = Metrics.recall_score(
        y_true=Array(y_true), y_predict=Array(y_predict), average="micro"
    )
    expected_recall_score_micro = recall_score(y_true, y_predict, average="micro")
    assert recall_score_micro, expected_recall_score_micro


def test_calculate_f1_score():
    y_true_binary = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_predict_binary = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]

    f1_score_binary = Metrics.f1_score(
        y_true=Array(y_true_binary), y_predict=Array(y_predict_binary)
    )
    expected_f1_score_binary = f1_score(y_true_binary, y_predict_binary)
    assert f1_score_binary, expected_f1_score_binary

    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    y_predict = [0, 0, 1, 1, 1, 2, 0, 1, 2, 1]
    f1_score_macro = Metrics.f1_score(
        y_true=Array(y_true), y_predict=Array(y_predict), average="macro"
    )
    expected_f1_score_macro = f1_score(y_true, y_predict, average="macro")
    assert f1_score_macro, expected_f1_score_macro

    f1_score_micro = Metrics.f1_score(
        y_true=Array(y_true), y_predict=Array(y_predict), average="micro"
    )
    expected_f1_score_micro = f1_score(y_true, y_predict, average="micro")
    assert f1_score_micro, expected_f1_score_micro
