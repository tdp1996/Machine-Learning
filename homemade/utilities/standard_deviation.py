import math
from typing import Union
from utilities.mean_median_mode import calculate_mean

def calculate_variance(data: list) -> Union[float,int]:
    """
    Variance is the measure of spread in data from its mean position
    """
    mean = calculate_mean(data)
    squared_diffs = [(value - mean)**2 for value in data]
    variance = sum(squared_diffs) / len(squared_diffs)
    return variance

def calculate_standard_deviation(data: list) -> Union[float,int]:
    """
    - Standard deviation is a number that describes how spread out the values are.
    - A low standard deviation means that most of the numbers are close to the mean (average) value.
    - A high standard deviation means that the values are spread out over a wider range.
    """
    variance = calculate_variance(data)
    std = math.sqrt(variance)
    return std

if __name__ == "__main__":
    speed = [86,87,88,86,87,85,86]
    print(calculate_standard_deviation(speed))
    
