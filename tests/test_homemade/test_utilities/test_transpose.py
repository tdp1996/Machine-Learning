import numpy as np
from homemade.utilities.transpose import transpose

def test_transpose():
    a = [[1, 2, 3], 
         [4, 5, 6]]
    np_a = np.array(a)

    transposed_a = transpose(a)
    expected_output = np.transpose(np_a)
    assert np.allclose(transposed_a,expected_output)
