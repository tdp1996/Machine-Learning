import math
import numpy
from homemade.utilities.standard_deviation import calculate_variance, calculate_standard_deviation

def test_calculate_variance():
    data = [32,111,138,28,59,77,97]
    variance = calculate_variance(data=data)
    np_variance = numpy.var(data)
    assert math.isclose(variance,np_variance)


def test_calculate_standard_deviation():
    data = [32,111,138,28,59,77,97]
    std = calculate_standard_deviation(data=data)
    np_std = numpy.std(data)
    assert math.isclose(std,np_std)