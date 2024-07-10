from typing import Union

def add(x1: list[list[Union[int, float]]], 
        x2: Union[int,float,list[Union[int,float]],list[list[Union[int, float]]]]) -> list[list[Union[int,float]]]:
    """
    Adds a scalar, vector, or matrix to another matrix.

    Args:
         x1: A 2D list (matrix) of integers or floats.
         x2: An integer, float, 1D list (vector), or 2D list (matrix) of integers or floats.

    Return: A 2D list (matrix) where x2 is added to x1.
    """

    if isinstance(x2,(int,float)):
        result = [[item[i] + x2 for i in range(len(item))] for item in x1]
    elif isinstance(x2,list) and all(isinstance(item,(int,float)) for item in x2):
        result = [[(x1[i][j] + x2[j]) for j in range(len(x2))] for i in range(len(x1))]
    else:
        num_rows = len(x1)
        num_cols = len(x1[0])
        if len(x2) != num_rows or any(len(row)!=num_cols for row in x2):
            raise ValueError("Two matrices must have the same dimensions") 
        result = [[x1[i][j] + x2[i][j] for j in range(num_cols)] for i in range(num_rows)]

    return result

    
def subtract(x1: list[list[Union[int, float]]], 
            x2: Union[int,float,list[Union[int,float]],list[list[Union[int, float]]]]) -> list[list[Union[int,float]]]:
    """
    Subtracts a scalar, vector, or matrix to another matrix.

    Args:
         x1: A 2D list (matrix) of integers or floats.
         x2: An integer, float, 1D list (vector), or 2D list (matrix) of integers or floats.

    Return: A 2D list (matrix) where x2 is subtracted to x1.
    """

    if isinstance(x2,(int,float)):
        result = [[item[i] - x2 for i in range(len(item))] for item in x1]
    elif isinstance(x2,list) and all(isinstance(item,(int,float)) for item in x2):
        result = [[(x1[i][j] - x2[j]) for j in range(len(x2))] for i in range(len(x1))]
    else:
        num_rows = len(x1)
        num_cols = len(x1[0])
        if len(x2) != num_rows or any(len(row)!=num_cols for row in x2):
            raise ValueError("Two matrices must have the same dimensions") 
        result = [[x1[i][j] - x2[i][j] for j in range(num_cols)] for i in range(num_rows)]

    return result
