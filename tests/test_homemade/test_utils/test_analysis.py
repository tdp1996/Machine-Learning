import pytest
import numpy as np
from homemade.utils.analysis import calculate_mean, calculate_variance, calculate_standard_deviation
from homemade.utils.classess import Array


def test_calculate_mean():
    a_1D = Array([99,86,87,88,111,86])
    np_1D = np.array([99,86,87,88,111,86])
    a_2D = Array([[99,86,87,88,111,86],[103,87,94,78,77,85]])
    np_2D = np.array([[99,86,87,88,111,86],[103,87,94,78,77,85]])

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
    with pytest.raises(
        ValueError, match=r"Axis 0 is not valid for array with shape"
    ):
        a_incompatible = [1, 2, 3]
        calculate_mean(Array(a_incompatible), axis=0)

    with pytest.raises(
        ValueError, match=r"Axis 1 is not valid for array with shape"
    ):
        a_incompatible = [1, 2, 3]
        calculate_mean(Array(a_incompatible), axis=1)

    with pytest.raises(
        ValueError, match=r"Invalid axis"
    ):
        a_incompatible = [[1, 2, 3], [4, 5, 6]]
        calculate_mean(Array(a_incompatible), axis=2)

    


# def test_calculate_variance():
#     data = [[99,86,87,88,111,86],[103,87,94,78,77,85]]
#     variance = calculate_variance(data=data,axis=1)
#     np_variance = np.var(data,axis=1)
#     assert np.allclose(variance,np_variance,atol=1e-8)


# def test_calculate_standard_deviation():
#     data = data = [[99,86,87,88,111,86],[103,87,94,78,77,85]]
#     std = calculate_standard_deviation(data=data)
#     np_std = np.std(data)
#     assert np.allclose(std,np_std,atol=1e-8)