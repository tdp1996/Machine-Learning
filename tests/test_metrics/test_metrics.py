import math
from metrics.metrics import r_square, accuracy, Mean_Squared_Error, Mean_Absolute_Error, Precision_score, Recall_score, F1_score
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, mean_absolute_error, precision_score,recall_score, f1_score


def test_r_square():
    y_true = [3,-0.5,2,7,4.2,1.5,3.3,2.8,1.2,4.6]
    y_predict = [2.5,0.0,2,8,4.0,1.7,3.5,2.6,1.4,4.8]
    output = r_square(y_true=y_true, y_predict=y_predict)
    expected_output = r2_score(y_true, y_predict)
    assert output == expected_output
              
def test_Mean_Squared_Error():
    y_true = [3,-0.5,2,7,4.2,1.5,3.3,2.8,1.2,4.6]
    y_predict = [2.5,0.0,2,8,4.0,1.7,3.5,2.6,1.4,4.8]
    mse = Mean_Squared_Error(y_true=y_true, y_predict=y_predict)
    expected_output = mean_squared_error(y_true,y_predict)
    assert math.isclose(mse,expected_output)

def test_Mean_Absolute_Error():
    y_true = [3,-0.5,2,7,4.2,1.5,3.3,2.8,1.2,4.6]
    y_predict = [2.5,0.0,2,8,4.0,1.7,3.5,2.6,1.4,4.8]
    mae = Mean_Absolute_Error(y_true=y_true, y_predict=y_predict)
    expected_output = mean_absolute_error(y_true, y_predict)
    assert math.isclose(mae,expected_output)

def test_accuracy():
    y_true = [1,0,1,1,0,1,0,0,1,0]
    y_predict = [1,0,1,0,0,1,0,1,1,0]
    score = accuracy(y_true=y_true, y_predict=y_predict)
    expected_score = accuracy_score(y_true, y_predict)
    assert score == expected_score


def test_precision():
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    y_predict = [0, 0, 1, 1, 1, 2, 0, 1, 2, 1]
    precision_score_macro = Precision_score(y_true=y_true, y_predict=y_predict, average="macro")
    expected_precision_score_macro = precision_score(y_true, y_predict, average="macro")

    precision_score_micro = Precision_score(y_true=y_true, y_predict=y_predict, average="micro")
    expected_precision_score_micro = precision_score(y_true, y_predict, average="micro")

    y_true_binary =  [1,0,1,1,0,1,0,0,1,0]
    y_predict_binary = [1,0,1,0,0,1,0,1,1,0]
    precision_score_binary = Precision_score(y_true=y_true_binary, y_predict=y_predict_binary)
    expected_precision_score_binary = precision_score(y_true_binary, y_predict_binary)

    assert math.isclose(precision_score_macro,expected_precision_score_macro)
    assert math.isclose(precision_score_micro,expected_precision_score_micro)
    assert math.isclose(precision_score_binary,expected_precision_score_binary)

def test_Recall_score():
    y_true_binary =  [1,0,1,1,0,1,0,0,1,0]
    y_predict_binary = [1,0,1,0,0,1,0,1,1,0]
    recall_score_binary = Recall_score(y_true=y_true_binary, y_predict=y_predict_binary)
    expected_recall_score_binary = recall_score(y_true_binary, y_predict_binary)
    
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    y_predict = [0, 0, 1, 1, 1, 2, 0, 1, 2, 1]
    recall_score_macro = Recall_score(y_true=y_true, y_predict=y_predict, average="macro")
    expected_recall_score_macro = recall_score(y_true, y_predict, average="macro")

    recall_score_micro = Recall_score(y_true=y_true, y_predict=y_predict, average="micro")
    expected_recall_score_micro = recall_score(y_true, y_predict, average="micro")

    assert math.isclose(recall_score_binary,expected_recall_score_binary)
    assert math.isclose(recall_score_macro, expected_recall_score_macro)
    assert math.isclose(recall_score_micro, expected_recall_score_micro)

def test_F1_score():
    y_true_binary =    [1,0,1,1,0,1,0,0,1,0]
    y_predict_binary = [1,0,1,0,0,1,0,1,1,0]
    f1_score_binary = F1_score(y_true=y_true_binary, y_predict=y_predict_binary)
    expected_f1_score_binary = f1_score(y_true_binary, y_predict_binary)
    
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    y_predict = [0, 0, 1, 1, 1, 2, 0, 1, 2, 1]
    f1_score_macro = F1_score(y_true=y_true, y_predict=y_predict, average="macro")
    expected_f1_score_macro = f1_score(y_true, y_predict, average="macro")

    f1_score_micro = F1_score(y_true=y_true, y_predict=y_predict, average="micro")
    expected_f1_score_micro = f1_score(y_true, y_predict, average="micro")

    assert math.isclose(f1_score_binary, expected_f1_score_binary)
    assert math.isclose(f1_score_macro, expected_f1_score_macro)
    assert math.isclose(f1_score_micro, expected_f1_score_micro)

