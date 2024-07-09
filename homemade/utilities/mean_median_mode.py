from typing import Union, Optional
import itertools


def calculate_mean(data: list[list[Union[int,float]]], axis: Optional[int]=None) -> list[Union[float,int]]:
    """
    Calculate the mean value(s) from a nested list of numerical data.

    Args:
    - data (list[list[Union[int,float]]]: A nested list containing numerical data.
    - axis (Optional[int]): Axis along which the mean is computed. If None (default), compute the mean of the flattened array.

    Returns:
    - Union[float, int]: The mean value(s) calculated based on the specified axis.

    Notes:
    - If axis is 0, computes the mean along rows.
    - If axis is 1, computes the mean along columns.
    - If axis is None or any other value, computes the mean of the flattened data.

    Example:
    >>> data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> calculate_mean(data, axis=0)
    [4.0, 5.0, 6.0]
    >>> calculate_mean(data, axis=1)
    [2.0, 5.0, 8.0]
    >>> calculate_mean(data)
    5.0
    """

    if axis==0:
        converted_data = [[item[i] for item in data] for i in range(len(data[0]))]
        mean = [sum(converted_data[i])/len(converted_data[i]) for i in range(len(converted_data))]
    elif axis==1:
        mean = [sum(data[i])/len(data[i]) for i in range(len(data))]
    else:
        sum_data = sum(itertools.chain.from_iterable(data))
        len_data = len(list(itertools.chain.from_iterable(data)))
        mean = sum_data/len_data
        
    return mean



def calculate_median(data: list[list[Union[int,float]]], axis: Optional[int]=None)  -> list[Union[float,int]]:
    """
    Calculate the median value(s) from a nested list of numerical data.

    Args:
    - data (list[list[Union[int,float]]]): A nested list containing numerical data.
    - axis (Optional[int]): Axis along which the median is computed. If None (default), compute the median of the flattened array.

    Returns:
    - list[Union[float, int]]: The median value(s) calculated based on the specified axis.

    Notes:
    - If axis is 0, computes the median along rows.
    - If axis is 1, computes the median along columns.
    - If axis is None or any other value, computes the median of the flattened data.

    Example:
    >>> data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> calculate_median(data, axis=0)
    [4.0, 5.0, 6.0]
    >>> calculate_median(data, axis=1)
    [2.0, 5.0, 8.0]
    >>> calculate_median(data)
    5.0
    """
    def compute(sorted_data, length):
        if length % 2 == 0:
            median = [(item[int(length/2) - 1] + item[int(length/2)]) / 2 for item in sorted_data]
        else:
            median = [item[length//2] for item in sorted_data]
        return median
    
    sorted_data = []

    if axis==None:
        converted_data = list(itertools.chain.from_iterable(data))
        sorted_data = [sorted(converted_data)]

    if axis==0:
        converted_data = [[item[i] for item in data] for i in range(len(data[0]))]
        sorted_data = [sorted(converted_data[i]) for i in range(len(converted_data))]
    
    if axis==1:
        sorted_data = [sorted(data[i]) for i in range(len(data))]

    length = len(sorted_data[0])
    median = compute(sorted_data,length)
    return median

