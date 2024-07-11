from typing import Union

def add_matrix(x1: list[list[Union[int, float]]], 
        x2: Union[int,float,list[Union[int,float]],list[list[Union[int, float]]]]) -> list[list[Union[int,float]]]:
    """
    Adds a scalar, vector, or matrix to another matrix.

    Args:
         x1: A 2D list (matrix) of integers or floats.
         x2: An integer, float, 1D list (vector), or 2D list (matrix) of integers or floats.

    Return: A 2D list (matrix) where x2 is added to x1.
    """
    #check if x2 is a scalar
    if isinstance(x2,(int,float)):
        result = [[item[i] + x2 for i in range(len(item))] for item in x1]
    
    #check if x2 is a vector
    elif isinstance(x2,list) and all(isinstance(item,(int,float)) for item in x2):
        if len(x1[0])!= len(x2):
            raise ValueError(f"operands could not be broadcast together with shapes {(len(x1), len(x1[0]))} {(len(x2),)}")
        result = [[(x1[i][j] + x2[j]) for j in range(len(x2))] for i in range(len(x1))]
    
    # Check if x2 is a matrix
    else:
        num_rows = len(x1)
        num_cols = len(x1[0])
        if len(x2) != num_rows or any(len(row)!=num_cols for row in x2):
            raise ValueError(f"two matrix must have the same dimesions {(num_rows, num_cols)} {(len(x2), len(x2[0]))}") 
        result = [[x1[i][j] + x2[i][j] for j in range(num_cols)] for i in range(num_rows)]

    return result

    
def subtract_matrix(x1: list[list[Union[int, float]]], 
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
        if len(x1[0])!= len(x2):
            raise ValueError(f"operands could not be broadcast together with shapes {(len(x1), len(x1[0]))} {(len(x2),)}")
        result = [[(x1[i][j] - x2[j]) for j in range(len(x2))] for i in range(len(x1))]
    else:
        num_rows = len(x1)
        num_cols = len(x1[0])
        if len(x2) != num_rows or any(len(row)!=num_cols for row in x2):
            raise ValueError(f"two matrix must have the same dimesions {(num_rows, num_cols)} {(len(x2), len(x2[0]))}") 
        result = [[x1[i][j] - x2[i][j] for j in range(num_cols)] for i in range(num_rows)]

    return result


def dot_matrix(x1: list[list[Union[int, float]]], 
                x2: Union[int,float,list[Union[int,float]],list[list[Union[int, float]]]]) -> list[list[Union[int,float]]]:
    """
    Perform the dot product of two matrices or a matrix and a scalar/vector.

    Args:
        x1 (list of list of Union[int, float]): The first matrix.
        x2 (Union[int, float, list of Union[int, float], list of list of Union[int, float]]): The second operand which can be 
        a scalar, vector, or matrix.

    Returns:
        list of list of Union[int, float]: The result of the dot product operation.

    Raises:
    ValueError: If the dimensions of the matrices are not compatible for multiplication.
    """
    
    if isinstance(x2,(int,float)):
        result = [[item[i] * x2 for i in range(len(item))] for item in x1]
    
    elif all(isinstance(item,(int,float)) for item in x2):
        if len(x1[0])!= len(x2):
             raise ValueError(f"shapes {(len(x1), len(x1[0]))} and {(len(x2),)} not aligned")
        result = [sum(item[i]*x2[i] for i in range(len(x2))) for item in x1]
            
    else:
        rows_x1 = len(x1)
        cols_x1 = len(x1[0])
        cols_x2 = len(x2[0])
        if rows_x1!=cols_x2:
            raise ValueError(f"shapes {(rows_x1, cols_x1)} and {(len(x2), len(x2[0]))} not algined")
        # Initialize the result matrix with zeros
        result = [[0 for _ in range(cols_x2)] for _ in range(rows_x1)]

        # Perform matrix multiplication
        for i in range(rows_x1):
            for j in range(cols_x2):
                for k in range(cols_x1):
                    result[i][j] += x1[i][k] * x2[k][j]

    return result
        