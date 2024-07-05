from typing import Optional, Union

def calculate_r_square(y_true: list[Union[float, int]], 
            y_predict: list[Union[float, int]]) ->float:
    """
    Calculate the R-squared (coefficient of determination) score.

    Args:
        y_true (list[Union[float, int]]): Actual target values.
        y_predict (list[Union[float, int]]): Predicted target values.

    Returns:
        float: R-squared score.
    """
    mean_y_true = sum(y_true)/len(y_true)
    numerator = sum((y_true[i] - y_predict[i]) ** 2 for i in range(len(y_true)))
    denominator = sum((y_true[i] - mean_y_true) ** 2 for i in range(len(y_true)))
    return 1 - (numerator/denominator)


def calculate_mean_squared_error(y_true: list[Union[float, int]], 
                    y_predict: list[Union[float, int]]) ->float:
    """
    Calculate the Mean Squared Error (MSE).

    Args:
        y_true (list[Union[float, int]]): Actual target values.
        y_predict (list[Union[float, int]]): Predicted target values.

    Returns:
        float: Mean Squared Error.
    """
    
    mse = sum((y_true[i] - y_predict[i])**2 for i in range(len(y_true))) / len(y_true)
    return mse


def calculate_mean_absolute_error(y_true: list[Union[float, int]], 
                        y_predict: list[Union[float, int]]) ->float:
    """
    Calculate the Mean Absolute Error (MAE).

    Args:
        y_true (list[Union[float, int]]): Actual target values.
        y_predict (list[Union[float, int]]): Predicted target values.

    Returns:
        float: Mean Absolute Error.
    """
    mae = sum(abs(y_true[i] - y_predict[i]) for i in range(len(y_true))) / len(y_true)
    return mae


def calculate_accuracy_score(y_true: list[Union[float, int]], 
            y_predict: list[Union[float, int]]) ->float:
    """
    Calculate the accuracy score.

    Args:
        y_true (list[Union[float, int]]): Actual target values.
        y_predict (list[Union[float, int]]): Predicted target values.

    Returns:
        float: Accuracy score.
    """
    count = 0
    for a,b in zip(y_true,y_predict):
        if a==b:
            count +=1
    return count/len(y_true)


def calculate_precision_score(y_true: list[Union[float, int]], 
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
    precision_scores_dict = {}
    precision_score = 0
    if average not in (None, "macro", "micro"):
        raise ValueError("Average must be one of None, 'macro', or 'micro'")
    
    classess = set(y_true)

    # calculate precision score for binary classification
    if average is None:
        TP = sum((1 for yt,yp in zip(y_true, y_predict) if yt==yp==1))
        FP = sum((1 for yt,yp in zip(y_true, y_predict) if yt==0 and yp==1))
        precision_score = TP/(TP+FP) if (TP+FP) > 0 else 0
    
    # calculate precision score for multiclass classification
    else:
        for c in classess:
            precision_c = f'precision_{c}'
            TP = sum((1 for yt,yp in zip(y_true, y_predict) if c==yt and c==yp))
            FP = sum((1 for yt,yp in zip(y_true, y_predict) if c!=yt and c==yp))
            precision_scores_dict[precision_c] = TP/(TP+FP) if (TP+FP) > 0 else 0

        # Macro-averaged precision                                           
        if average == "macro":          
            precision_score = sum(precision_scores_dict.values())/len(classess)
        
        # Micro-averaged precision
        elif average == "micro":   
            TP = sum((1 for yt,yp in zip(y_true, y_predict) if yt==yp))
            FP = sum((1 for yt,yp in zip(y_true, y_predict) if yt!=yp))
            precision_score = TP/(TP+FP) if (TP+FP) > 0 else 0
  
    return precision_score


def calculate_recall_score(y_true: list[Union[float, int]], 
            y_predict: list[Union[float, int]], 
            average: Optional[str] = None) -> float:
    """
    Calculate the recall for the given true labels and predicted labels.

    Args:
        y_true (List[Union[float, int]]): True labels.
        y_predict (List[Union[float, int]]): Predicted labels.
        average (Optional[str]): Type of averaging performed on the data. 
                                 "macro" or "micro" or None (default is None).

    Returns:
        float: Calculated recall score.
    """

    recall_score = 0
    recall_scores_dict = {}

    if average not in (None, "macro", "micro"):
        raise ValueError("Average must be one of None, 'macro', or 'micro'")
    
    classess = set(y_true)

    # calculate recall score for binary classification
    if average is None:
        TP = sum((1 for yt,yp in zip(y_true, y_predict) if yt==yp==1))
        FN = sum((1 for yt,yp in zip(y_true, y_predict) if yt==1 and yp==0))
        recall_score = TP/(TP+FN) if (TP+FN) > 0 else 0

    # calculate recall score for multiclass classification    
    else:
        for c in classess:
            recall_c = f'recall_{c}'
            TP = sum((1 for yt,yp in zip(y_true, y_predict) if c==yt and c==yp))
            FN = sum((1 for yt,yp in zip(y_true, y_predict) if c==yt and c!=yp))
            recall_scores_dict[recall_c] = TP/(TP+FN) if (TP+FN) > 0 else 0

        # Macro-averaged recall
        if average == "macro":
            recall_score = sum(recall_scores_dict.values()) / len(classess)

        # Micro-averaged recall
        elif average == "micro":
            TP = sum((1 for yt,yp in zip(y_true, y_predict) if yt==yp))
            FN = sum((1 for yt,yp in zip(y_true, y_predict) if yt!=yp))
            recall_score = TP/(TP+FN) if (TP+FN) > 0 else 0
            
    return recall_score



def calculate_f1_score(y_true: list[Union[float, int]], 
            y_predict: list[Union[float, int]], 
            average: Optional[str] = None) -> float:
    """
    Calculate the F1 score for the given true labels and predicted labels.

    Args:
        y_true (list[Union[float, int]]): True labels.
        y_predict (list[Union[float, int]]): Predicted labels.
        average (Optional[str]): Type of averaging performed on the data. 
                                 "macro", "micro" or None (default is None).

#     Returns:
#         float: Calculated F1 score.
#     """

    
    if average not in (None, "macro", "micro"):
        raise ValueError("Average must be one of None, 'macro', or 'micro'")

    # calculate precision score for binary classification
    if average is None:
        TP = sum((1 for yt,yp in zip(y_true, y_predict) if yt==yp==1))
        FP = sum((1 for yt,yp in zip(y_true, y_predict) if yt==0 and yp==1))
        FN = sum((1 for yt,yp in zip(y_true, y_predict) if yt==1 and yp==0))
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0
        recall = TP/(TP+FN) if (TP+FN) > 0 else 0
        f1_score = 2 * ((precision * recall) / (precision + recall))

    # calculate precision score for multiclass classification
    else:
        classess = set(y_true)
        precision_score = []
        recall_score = []
        sum_TP = 0
        sum_FP = 0
        sum_FN = 0

        for c in classess:
            TP = sum((1 for yt,yp in zip(y_true, y_predict) if c==yt and c==yp))
            FP = sum((1 for yt,yp in zip(y_true, y_predict) if c!=yt and c==yp))
            FN = sum((1 for yt,yp in zip(y_true, y_predict) if c==yt and c!=yp))
            precision_score.append( TP/(TP+FP) if (TP+FN) > 0 else 0)
            recall_score.append(TP/(TP+FN) if (TP+FN) > 0 else 0)
            sum_TP +=TP
            sum_FP +=FP
            sum_FN +=FN

        if average == "macro":
            f1_score = sum(2*(p * r)/(p+r) if (p+r)!=0 else 0 for p,r in zip(precision_score,recall_score)) / len(classess)
    
        if average == "micro":
            precision = sum_TP / (sum_TP + sum_FP) if (sum_TP + sum_FP)!=0 else 0
            recall = sum_TP / (sum_TP + sum_FN) if (sum_TP + sum_FN)!=0 else 0
            f1_score = 2*(precision*recall)/(precision+recall) if (precision+recall)!=0 else 0
    
    return f1_score

    