import numpy as np
import pytest
import re
from homemade.utils.operations import add_matrix, subtract_matrix, dot_matrix


def test_add_matrix():
    x1 = [[1, 2, 3], 
         [4, 5, 6]]
    x2 = 3
    x3 = [0.5, 2, -3]
    x4 = [[10, 20, 30],
          [40, 50, 60]]
    
    output1 = add_matrix(x1=x1,x2=x2)
    expected_output1 = np.add(x1,x2)
    assert np.allclose(output1,expected_output1)

    output2 = add_matrix(x1=x1,x2=x3)
    expected_output2 = np.add(x1,x3)
    assert np.allclose(output2,expected_output2)

    output3 = add_matrix(x1=x1,x2=x4)
    expected_output3 = np.add(x1,x4)
    assert np.allclose(output3,expected_output3)

def test_add_matrix_error():
    x1 = [[1, 2, 3], 
         [4, 5, 6]]
    x2 = [0.5, -1]
    x3 = [[10, 20],
          [40, 50]]
    with pytest.raises(
            ValueError,
            match=re.escape("Cannot add a vector of length 2 to a matrix with 3 columns.")):
            add_matrix(x1=x1, x2=x2)

    with pytest.raises(
        ValueError,
        match=re.escape("Cannot add a matrix of shape (2, 2) to a matrix of shape (2, 3).")):
        add_matrix(x1=x1, x2=x3)
      
      

def test_subtract_matrix():
    x1 = [[1, 2, 3], 
         [4, 5, 6]]
    x2 = 3
    x3 = [0.5, 2, -3]
    x4 = [[10, 20, 30],
          [40, 50, 60]]
    
    output1 = subtract_matrix(x1=x1,x2=x2)
    expected_output1 = np.subtract(x1,x2)
    assert np.allclose(output1,expected_output1)

    output2 = subtract_matrix(x1=x1,x2=x3)
    expected_output2 = np.subtract(x1,x3)
    assert np.allclose(output2,expected_output2)

    output3 = subtract_matrix(x1=x1,x2=x4)
    expected_output3 = np.subtract(x1,x4)
    assert np.allclose(output3,expected_output3)

def test_subtract_matrix_error():
    x1 = [[1, 2, 3], 
         [4, 5, 6]]
    x2 = [0.5, -1]
    x3 = [[10, 20],
          [40, 50]]
    with pytest.raises(
            ValueError,
            match=re.escape("Cannot subtract a vector of length 2 to a matrix with 3 columns.")):
            subtract_matrix(x1=x1, x2=x2)

    with pytest.raises(
        ValueError,
        match=re.escape("Cannot subtract a matrix of shape (2, 2) to a matrix of shape (2, 3).")):
        subtract_matrix(x1=x1, x2=x3)

def test_dot_matrix():
    x1 = [[1, 2, 3],
          [4, 5, 6]]
    x2 = 3
    x3 = [-2, 0.5, 1.5]
    x4 = [[7, 8], [9, 10], [11, 12]]

    output1 = dot_matrix(x1=x1, x2=x2)
    expected_output1 = np.dot(x1,x2)
    assert np.allclose(output1,expected_output1)

    output2 = dot_matrix(x1=x1,x2=x3)
    expected_output2 = np.dot(x1,x3)
    assert np.allclose(output2,expected_output2)

    output3 = dot_matrix(x1=x1,x2=x4)
    expected_output3 = np.dot(x1,x4)
    assert np.allclose(output3,expected_output3)
    
def test_dot_matrix_error():
    x1 = [[1, 2, 3], 
         [4, 5, 6]]
    x2 = [0.5, -1]
    x3 = [[10, 20, 30],
          [40, 50, 30]]
    with pytest.raises(
            ValueError,
            match=re.escape("shapes (2, 3) and (2,) not aligned for vector multiplication.")):
            dot_matrix(x1=x1, x2=x2)

    with pytest.raises(
        ValueError,
        match=re.escape("shapes (2, 3) and (2, 3) not aligned for matrix multiplication.")):
        dot_matrix(x1=x1, x2=x3)