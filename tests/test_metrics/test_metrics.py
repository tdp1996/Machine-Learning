from metrics.metrics import r_square, accuracy, F1_Score
from sklearn.metrics import r2_score, accuracy_score, f1_score


def test_r_square():
    y_true = [3,-0.5,2,7,4.2,1.5,3.3,2.8,1.2,4.6]
    y_predict = [2.5,0.0,2,8,4.0,1.7,3.5,2.6,1.4,4.8]
    output = r_square(y_true=y_true, y_predict=y_predict)
    expected_output = r2_score(y_true, y_predict)
    assert output == expected_output
              

def test_accuracy():
    y_true = [1,0,1,1,0,1,0,0,1,0]
    y_predict = [1,0,1,0,0,1,0,1,1,0]
    score = accuracy(y_true=y_true, y_predict=y_predict)
    expected_score = accuracy_score(y_true, y_predict)
    assert score == expected_score

# def test_F1_Score():
#     y_true = [1,0,1,1,0,1,0,0,1,0]
#     y_predict = [1,0,1,0,0,1,0,1,1,0]
#     score = F1_Score(y_true=y_true, y_predict=y_predict)
#     expected_score = f1_score(y_true, y_predict)
#     assert score == expected_score

