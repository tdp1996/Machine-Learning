import math
import numpy
from typing import Union
from scipy import stats
from utilities.mean_median_mode import calculate_mean, calculate_median, calculate_mode

def test_calculate_mean():
    data = [99,86,87,88,111,86,103,87,94,78,77,85,86]
    mean = calculate_mean(data=data)
    np_mean = numpy.mean(data)
    assert isinstance(mean,Union[float,int])
    assert math.isclose(mean,np_mean)


def test_calculate_median():
    data = [99,86,87,88,111,86,103,87,94,78,77,85,86]
    median = calculate_median(data=data)
    np_median = numpy.median(data)
    assert math.isclose(median,np_median)
    assert median == 87


def test_calculate_mode():
    data = [99,86,87,88,111,86,103,87,94,78,77,85,86]
    mode = calculate_mode(data=data)
    np_mode = stats.mode(data)
    assert math.isclose(mode,np_mode.mode)