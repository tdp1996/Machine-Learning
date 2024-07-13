from typing import Union, Optional
from .classess import Array


def calculate_mean(data: Array, axis: Optional[int] = None) -> Array:
    """
    Calculate the mean value(s) from a nested list of numerical data.

    Args:
    - data (Array): An Array object containing numerical data.
    - axis (Optional[int]): Axis along which the mean is computed. If None (default), compute the mean of the flattened array.

    Returns:
    - Array: The mean value(s) calculated based on the specified axis.

    Notes:
    - If axis is 0, computes the mean along rows.
    - If axis is 1, computes the mean along columns.
    - If axis is None or any other value, computes the mean of the flattened data.

    Example:
    >>> data = Array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> calculate_mean(data, axis=0)
    Array([4.0, 5.0, 6.0])
    >>> calculate_mean(data, axis=1)
    Array([2.0, 5.0, 8.0])
    >>> calculate_mean(data)
    5.0
    """
    if axis is None:
        total_sum = data.sum()
        total_elements = data.shape[0] if len(data.shape) == 1 else data.shape[0] * data.shape[1]
        return total_sum / total_elements
    elif axis == 0:
        if len(data.shape) != 2:
            raise ValueError(f"Axis 0 is not valid for array with shape {data.shape}")
        return data.sum(axis=0) / data.shape[0]
    elif axis == 1:
        if len(data.shape) != 2:
            raise ValueError(f"Axis 1 is not valid for array with shape {data.shape}")
        return data.sum(axis=1) / data.shape[1]
    else:
        raise ValueError(f"Invalid axis")
        


def calculate_variance(data: Array, axis: Optional[int] = None) -> Array:
    """
    Calculate the variance from an Array of numerical data.

    Args:
        data (Array): An Array object containing numerical data.
        axis (Optional[int]): Axis along which the variance is computed. 
                              If None (default), compute the variance of the flattened array.

    Returns:
        Array: The variance value(s) calculated based on the specified axis.

    Raises:
        ValueError: If the shapes of the arrays are not compatible for the operation.

    Example:
    >>> data = Array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> calculate_variance(data, axis=0)
    Array([6.0, 6.0, 6.0])
    >>> calculate_variance(data, axis=1)
    Array([0.6666666666666666, 0.6666666666666666, 0.6666666666666666])
    >>> calculate_variance(data)
    6.666666666666667
    """
    mean_data = calculate_mean(data, axis=axis)

    if axis is None:
        if len(data.shape) == 1:
            squared_diff = Array([(x - mean_data) ** 2 for x in data.data])
        elif len(data.shape) == 2:
            squared_diff = Array([[(x - mean_data) ** 2 for x in row] for row in data.data])
        total_elements = data.shape[0] if len(data.shape) == 1 else data.shape[0] * data.shape[1]
        return squared_diff.sum() / total_elements

    elif axis == 0:
        if len(data.shape) != 2:
            raise ValueError(f"Axis 0 is not valid for array with shape {data.shape}")
        squared_diff = Array([[(data.data[i][j] - mean_data.data[j]) ** 2 for j in range(data.shape[1])] for i in range(data.shape[0])])
        return squared_diff.sum(axis=0) / data.shape[0]

    elif axis == 1:
        if len(data.shape) != 2:
            raise ValueError(f"Axis 1 is not valid for array with shape {data.shape}")
        squared_diff = Array([[(data.data[i][j] - mean_data.data[i]) ** 2 for j in range(data.shape[1])] for i in range(data.shape[0])])
        return squared_diff.sum(axis=1) / data.shape[1]

    else:
        raise ValueError("Invalid axis")

    

# def calculate_standard_deviation(data: list[list[Union[int,float]]], axis: Optional[int]=None) -> Union[float,int]:
#     """
#     - Standard deviation is a number that describes how spread out the values are.
#     - A low standard deviation means that most of the numbers are close to the mean (average) value.
#     - A high standard deviation means that the values are spread out over a wider range.
#     """
#     variance = calculate_variance(data,axis)
#     if axis==0 or axis==1:
#         std = [math.sqrt(variance[i]) for i in range(len(variance))]
#     else:
#         std = math.sqrt(variance)
#     return std

# if __name__ == "__main__":
#     speed = [86,87,88,86,87,85,86]
#     print(calculate_standard_deviation(speed))
    
