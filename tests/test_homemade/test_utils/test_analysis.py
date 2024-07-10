import numpy as np
from homemade.utils.analysis import calculate_mean, calculate_variance, calculate_standard_deviation


def test_calculate_mean():
    data = [[99,86,87,88,111,86,103,87,94,78,77,85]]
    np_array = np.array(data)
    mean = calculate_mean(data=data,axis=1)
    np_mean = np.mean(np_array,axis=1)
    assert np.allclose(mean, np_mean, atol=1e-8)


def test_calculate_variance():
    data = [[99,86,87,88,111,86],[103,87,94,78,77,85]]
    variance = calculate_variance(data=data,axis=1)
    np_variance = np.var(data,axis=1)
    assert np.allclose(variance,np_variance,atol=1e-8)


def test_calculate_standard_deviation():
    data = data = [[99,86,87,88,111,86],[103,87,94,78,77,85]]
    std = calculate_standard_deviation(data=data)
    np_std = np.std(data)
    assert np.allclose(std,np_std,atol=1e-8)