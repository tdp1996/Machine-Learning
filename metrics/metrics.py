def r_square(y_true: list, y_predict: list) ->float:
    mean_y_true = sum(y_true)/len(y_true)
    numerator = sum((y_true[i] - y_predict[i]) ** 2 for i in range(len(y_true)))
    denominator = sum((y_true[i] - mean_y_true) ** 2 for i in range(len(y_true)))
    return 1 - (numerator/denominator)

def accuracy(y_true: list, y_predict: list) ->float:
    count = 0
    for a,b in zip(y_true,y_predict):
        if a==b:
            count +=1
    return count/len(y_true)

# def F1_Score(y_true: list, y_predict: list) ->float:
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 1
#     for a,b in zip(y_true, y_predict):
#         if a==1 and 
    