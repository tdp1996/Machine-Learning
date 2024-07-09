import numpy as np
from homemade.utilities.mean_median import calculate_mean, calculate_median

def test_calculate_mean():
    data = [[99,86,87,88,111,86,103,87,94,78,77,85]]
    np_array = np.array(data)
    mean = calculate_mean(data=data,axis=1)
    np_mean = np.mean(np_array,axis=1)
    assert np.allclose(mean, np_mean, atol=1e-8)



def test_calculate_median():
    data = [[99,86,87,88,111,86],[103,99,94,78,77,85]]
    median = calculate_median(data=data)
    np_median = np.median(data)
    assert np.allclose(median, np_median, atol=1e-8)

