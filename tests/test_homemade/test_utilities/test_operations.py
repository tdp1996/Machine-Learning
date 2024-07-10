import numpy as np
from homemade.utilities.operations import add_matrix, subtract_matrix


def test_add_matrix():
    x1 = [[1, 2, 3], 
         [4, 5, 6]]
    x2 = 3
    x3 = [0.5, 2, -3]
    x4 = [[10, 20, 30],
          [40, 50, 60]]
    
    output1 = add_matrix(x1,x2)
    expected_output1 = np.add(x1,x2)
    assert np.allclose(output1,expected_output1)

    output2 = add_matrix(x1,x3)
    expected_output2 = np.add(x1,x3)
    assert np.allclose(output2,expected_output2)

    output3 = add_matrix(x1,x4)
    expected_output3 = np.add(x1,x4)
    assert np.allclose(output3,expected_output3)

def test_subtract_matrix():
    x1 = [[1, 2, 3], 
         [4, 5, 6]]
    x2 = 3
    x3 = [0.5, 2, -3]
    x4 = [[10, 20, 30],
          [40, 50, 60]]
    
    output1 = subtract_matrix(x1,x2)
    expected_output1 = np.subtract(x1,x2)
    assert np.allclose(output1,expected_output1)

    output2 = subtract_matrix(x1,x3)
    expected_output2 = np.subtract(x1,x3)
    assert np.allclose(output2,expected_output2)

    output3 = subtract_matrix(x1,x4)
    expected_output3 = np.subtract(x1,x4)
    assert np.allclose(output3,expected_output3)
    