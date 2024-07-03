import math
from typing import Optional, Union


def r_square(y_true: list[Union[float, int]], 
            y_predict: list[Union[float, int]]) ->float:
    mean_y_true = sum(y_true)/len(y_true)
    numerator = sum((y_true[i] - y_predict[i]) ** 2 for i in range(len(y_true)))
    denominator = sum((y_true[i] - mean_y_true) ** 2 for i in range(len(y_true)))
    return 1 - (numerator/denominator)


def Mean_Squared_Error(y_true: list[Union[float, int]], 
                    y_predict: list[Union[float, int]]) ->float:
    mse = sum((y_true[i] - y_predict[i])**2 for i in range(len(y_true))) / len(y_true)
    return mse


def Mean_Absolute_Error(y_true: list[Union[float, int]], 
                        y_predict: list[Union[float, int]]) ->float:
    mae = sum(abs(y_true[i] - y_predict[i]) for i in range(len(y_true))) / len(y_true)
    return mae


def accuracy(y_true: list[Union[float, int]], 
            y_predict: list[Union[float, int]]) ->float:
    count = 0
    for a,b in zip(y_true,y_predict):
        if a==b:
            count +=1
    return count/len(y_true)


def precision(y_true: list[Union[float, int]], 
              y_predict: list[Union[float, int]], 
              average: Optional[str] = None) -> float:
    """
    Calculate the precision for the given true labels and predicted labels.

    Args:
        y_true (List[Union[float, int]]): True labels.
        y_predict (List[Union[float, int]]): Predicted labels.
        average (Optional[str]): Type of averaging performed on the data. 
                                 "macro" or "micro" or None (default is None).

    Returns:
        float: Calculated precision.
    """
    precision_scores = {}
    if average not in (None, "macro", "micro"):
        raise ValueError("Average must be one of None, 'macro', or 'micro'")
    
    classess = set(y_true)
    # calculate precision score for binary classification
    if average is None:
        TP = sum((1 for yt,yp in zip(y_true, y_predict) if yt==yp==1))
        FP = sum((1 for yt,yp in zip(y_true, y_predict) if yt==0 and yp==1))
        precision_score = TP/(TP+FP) if (TP+FP) > 0 else 0
    
    else:
        for c in classess:
            precision_c = f'precision_{c}'
            TP = sum((1 for yt,yp in zip(y_true, y_predict) if c==yt and c==yp))
            FP = sum((1 for yt,yp in zip(y_true, y_predict) if c!=yt and c==yp))
            precision_scores[precision_c] = TP/(TP+FP) if (TP+FP) > 0 else 0
                                                    
        if average == "macro":
            # Macro-averaged precision
            precision_score = sum(precision_scores.values())/len(classess)
  
    return precision_score

    



# def F1_Score(y_true: list, y_predict: list) ->float:
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 1
#     for a,b in zip(y_true, y_predict):
#         if a==1 and 
    