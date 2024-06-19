from typing import Union

def calculate_mean(data: list) -> Union[float,int]:
    """
    The mean value is the average value.
    To calculate the mean, find the sum of all values, and divide the sum by the number of values
    """
    sum_values = 0
    for value in data:
        sum_values += value
    mean = sum_values/len(data)
    return mean

def calculate_median(data: list)  -> Union[float,int]:
    """
    The median value is the value in the middle, after you have sorted all the values
    """
    sorted_data = sorted(data)
    length = len(sorted_data)
    if length % 2 == 0:
        median = (sorted_data[int(length/2) - 1] + sorted_data[int(length/2)]) / 2
    else:
        median = sorted_data[length//2]
    return median

def calculate_mode(data: list):
    """
    The Mode value is the value that appears the most number of times
    """
    frequency_dict = {}
    for item in data:
        if item in frequency_dict:
            frequency_dict[item]+=1
        else:
            frequency_dict[item]=1
    max_count = 0
    mode = None
    for key, value in frequency_dict.items():
        if value > max_count:
            max_count = value
            mode = key
    modes = [key for key, count in frequency_dict.items() if count == max_count]

    if len(modes) > 1:
        return modes
    else:
        return mode


if __name__ == "__main__":
    data = [1, 2, 2, 3, 3, 3,3, 4, 4, 4, 4, 5]
    mode = calculate_mode(data)
    print(f"The mode is: {mode}")
