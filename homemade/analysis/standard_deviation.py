import math
from typing import Union, Optional
import itertools
from .mean_median import calculate_mean

def calculate_variance(data: list[list[Union[int,float]]], axis: Optional[int]=None) -> Union[float,int]:
    """
    Calculate the variance of the data along the specified axis.
    
    Variance is a measure of the spread of data points from their mean position.

    Args:
        - data (list[list[Union[int,float]]]: A nested list containing numerical data.
        - axis (Optional[int]): Axis along which the variance is computed. If None (default), compute the variance of the flattened array.
    
    Returns:
        Union[float, int]: The variance value(s).

    Notes:
    - If axis is 0, computes the variance along rows.
    - If axis is 1, computes the variance along columns.
    - If axis is None or any other value, computes the variance of the flattened data.

    Example:
    >>> data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> calculate_variance(data, axis=0)
    [6.0, 6.0, 6.0]
    >>> calculate_variance(data, axis=1)
    [0.67, 0.67, 0.67]
    >>> calculate_variance(data)
    6.67
    """
    mean = calculate_mean(data,axis)
    variance = []
    
    if axis==0:
        flattened_data = [[item[i] for item in data] for i in range(len(data[0]))]
        for subdata, item in zip(flattened_data,mean):
            variance_i = sum((subdata[i] - item)**2 for i in range(len(subdata))) / len(subdata)
            variance.append(variance_i)
    elif axis==1:
        for subdata, item in zip(data,mean):
            variance_i = sum((subdata[i] - item)**2 for i in range(len(subdata))) / len(subdata)
            variance.append(variance_i)
    else:
        flattened_data = list(itertools.chain.from_iterable(data))
        squared_diffs = [(value - mean)**2 for value in flattened_data]
        variance = sum(squared_diffs) / len(squared_diffs)

    return variance

def calculate_standard_deviation(data: list[list[Union[int,float]]], axis: Optional[int]=None) -> Union[float,int]:
    """
    - Standard deviation is a number that describes how spread out the values are.
    - A low standard deviation means that most of the numbers are close to the mean (average) value.
    - A high standard deviation means that the values are spread out over a wider range.
    """
    variance = calculate_variance(data,axis)
    if axis==0 or axis==1:
        std = [math.sqrt(variance[i]) for i in range(len(variance))]
    else:
        std = math.sqrt(variance)
    return std

if __name__ == "__main__":
    speed = [86,87,88,86,87,85,86]
    print(calculate_standard_deviation(speed))
    
