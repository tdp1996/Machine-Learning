import pytest
import numpy as np
from homemade.utils.analysis import (
    calculate_mean,
    calculate_variance,
    calculate_standard_deviation,
)
from homemade.utils.classess import Array


def test_calculate_mean():
    a_1D = Array([1, 2, 3])
    np_1D = np.array([1, 2, 3])
    a_2D = Array([[1, 2, 3], [4, 5, 6]])
    np_2D = np.array([[1, 2, 3], [4, 5, 6]])

    # test mean of array 1D
    mean1 = calculate_mean(data=a_1D)
    np_mean1 = np.mean(np_1D)
    assert mean1 == np_mean1

    # test mean of array 2D
    mean2 = calculate_mean(data=a_2D)
    np_mean2 = np.mean(np_2D)
    assert mean2 == np_mean2

    # test mean of array 2D along axis 0
    mean3 = calculate_mean(data=a_2D, axis=0)
    np_mean3 = np.mean(np_2D, axis=0)
    assert mean3.tolist() == np_mean3.tolist()

    # test mean of array 2D along axis 1
    mean4 = calculate_mean(data=a_2D, axis=1)
    np_mean4 = np.mean(np_2D, axis=1)
    assert mean4.tolist() == np_mean4.tolist()


def test_calculate_mean_error():
    with pytest.raises(ValueError, match=r"Axis 1 is not valid for array with shape"):
        a_incompatible = [1, 2, 3]
        calculate_mean(Array(a_incompatible), axis=1)


def test_calculate_variance():
    a_1D = Array([1, 2, 3])
    np_1D = np.array([1, 2, 3])
    a_2D = Array([[1, 2, 3], [4, 5, 6]])
    np_2D = np.array([[1, 2, 3], [4, 5, 6]])

    # test mean of array 1D
    variance1 = calculate_variance(data=a_1D)
    np_variance1 = np.var(np_1D)
    assert variance1 == np_variance1

    # test mean of array 2D
    variance2 = calculate_variance(data=a_2D)
    np_variance2 = np.var(np_2D)
    assert variance2 == np_variance2

    # test mean of array 2D along axis 0
    variance3 = calculate_variance(data=a_2D, axis=0)
    np_variance3 = np.var(np_2D, axis=0)
    assert variance3.tolist() == np_variance3.tolist()

    # test mean of array 2D along axis 1
    variance4 = calculate_variance(data=a_2D, axis=1)
    np_variance4 = np.var(np_2D, axis=1)
    assert variance4.tolist() == np_variance4.tolist()

    variance5 = calculate_variance(data=a_1D, axis=0)
    np_variance5 = np.var(np_1D, axis=0)
    assert variance5 == np_variance5


def test_calculate_standard_deviation():
    a_1D = Array([1, 2, 3])
    np_1D = np.array([1, 2, 3])
    a_2D = Array([[1, 2, 3], [4, 5, 6]])
    np_2D = np.array([[1, 2, 3], [4, 5, 6]])

    # test mean of array 1D
    std1 = calculate_standard_deviation(data=a_1D)
    np_std1 = np.std(np_1D)
    assert std1 == np_std1

    # test mean of array 2D
    std1 = calculate_standard_deviation(data=a_2D)
    np_std2 = np.std(np_2D)
    assert std1 == np_std2

    # test mean of array 2D along axis 0
    std1 = calculate_standard_deviation(data=a_2D, axis=0)
    np_std3 = np.std(np_2D, axis=0)
    assert std1.tolist() == np_std3.tolist()

    # test mean of array 2D along axis 1
    std4 = calculate_standard_deviation(data=a_2D, axis=1)
    np_std4 = np.std(np_2D, axis=1)
    assert std4.tolist() == np_std4.tolist()
