import numpy as np
import re
import pytest
from homemade.utils.array import Array

def test_add_Array():
    a_2D = [[1, 2, 3],      
         [4, 5, 6]]
    a_1D = [-1, 0.5, 2]
    scalar = 3

    # test add scalar to array 2D
    output1 = Array(a_2D) + scalar
    expected_output1 = np.add(a_2D,scalar)
    assert output1.tolist() == expected_output1.tolist()

    #test add scalar to array 1D
    output2 = Array(a_1D) + scalar
    expected_output2 = np.add(a_1D,scalar)
    assert output2.tolist() == expected_output2.tolist()

    # test add array 1D to array 1D
    output3 = Array(a_1D) + Array(a_1D)
    expected_output3 = np.add(a_1D,a_1D)
    assert output3.tolist() == expected_output3.tolist()

    # test add an array 2D to an array 2D
    output4 = Array(a_2D) + Array(a_2D)
    expected_output4 = np.add(a_2D,a_2D)
    assert output4.tolist() == expected_output4.tolist()

    # test add an array 1D to an array 2D
    output5 =Array(a_1D) + Array(a_2D)
    expected_output5 = np.add(a_1D,a_2D)
    assert output5.tolist() == expected_output5.tolist()

def test_add_array_error():
    with pytest.raises(ValueError, match=r"operands could not be broadcast together with shapes"):
        a_2D_incompatible = [[1, 2], [3, 4]]
        a_1D_incompatible = [1, 2, 3]
        Array(a_2D_incompatible) + Array(a_1D_incompatible)

    with pytest.raises(ValueError, match=r"operands could not be broadcast together with shapes"):
        a_2D = [[1, 2, 3], [4, 5, 6]]
        a_2D_incompatible = [[1, 2], [3, 4]]
        Array(a_2D) + Array(a_2D_incompatible)

def test_sub_array():
    a_2D = [[1, 2, 3],      
         [4, 5, 6]]
    a_1D = [-1, 0.5, 2]
    scalar = 3

    # test subtract scalar from array 2D
    output1 = Array(a_2D) - scalar
    expected_output1 = np.subtract(a_2D,scalar)
    assert output1.tolist() == expected_output1.tolist()

    #test add scalar from array 1D
    output2 = Array(a_1D) - scalar
    expected_output2 = np.subtract(a_1D,scalar)
    assert output2.tolist() == expected_output2.tolist()

    # test subtract array 1D from array 1D
    output3 = Array(a_1D) - Array(a_1D)
    expected_output3 = np.subtract(a_1D,a_1D)
    assert output3.tolist() == expected_output3.tolist()

    # test subtract an array 2D from an array 2D
    output4 = Array(a_2D) - Array(a_2D)
    expected_output4 = np.subtract(a_2D,a_2D)
    assert output4.tolist() == expected_output4.tolist()

    # test subtract an array 1D from array 2D
    output5 = Array(a_2D) - Array(a_1D)
    expected_output5 = np.subtract(a_2D,a_1D)
    assert output5.tolist() == expected_output5.tolist()

def test_subtract_array_error():
    with pytest.raises(ValueError, match=r"operands could not be broadcast together with shapes"):
        a_2D_incompatible = [[1, 2], [3, 4]]
        a_1D_incompatible = [1, 2, 3]
        Array(a_2D_incompatible) - Array(a_1D_incompatible)

    with pytest.raises(ValueError, match=r"operands could not be broadcast together with shapes"):
        a_2D = [[1, 2, 3], [4, 5, 6]]
        a_2D_incompatible = [[1, 2], [3, 4]]
        Array(a_2D) - Array(a_2D_incompatible)

def test_multiply_array():
    a_2D = [[1, 2, 3],      
         [4, 5, 6]]
    a_1D = [-1, 0.5, 2]
    scalar = 3

    # test multiply an array 2D with a scalar
    output1 = Array(a_2D) * scalar
    expected_output1 = np.multiply(a_2D,scalar)
    assert output1.tolist() == expected_output1.tolist()

    # test multiply an array 1D with a scalar
    output2 = Array(a_1D) * scalar
    expected_output2 = np.multiply(a_1D,scalar)
    assert output2.tolist() == expected_output2.tolist()

    # test multiply two arrays 1D
    output3 = Array(a_1D) * Array(a_1D)
    expected_output3 = np.multiply(a_1D,a_1D)
    assert output3.tolist() == expected_output3.tolist()

    # test multiply two arrays 2D
    output4 = Array(a_2D) * Array(a_2D)
    expected_output4 = np.multiply(a_2D,a_2D)
    assert output4.tolist() == expected_output4.tolist()

    # test multiply an array 2D with an array 1D
    output5 = Array(a_2D) * Array(a_1D)
    expected_output5 = np.multiply(a_2D,a_1D)
    assert output5.tolist() == expected_output5.tolist()

def test_multiply_array_error():
    with pytest.raises(ValueError, match=r"operands could not be broadcast together with shapes"):
        a_2D_incompatible = [[1, 2], [3, 4]]
        a_1D_incompatible = [1, 2, 3]
        Array(a_2D_incompatible) * Array(a_1D_incompatible)

    with pytest.raises(ValueError, match=r"operands could not be broadcast together with shapes"):
        a_2D = [[1, 2, 3], [4, 5, 6]]
        a_2D_incompatible = [[1, 2], [3, 4]]
        Array(a_2D) * Array(a_2D_incompatible)

def test_dot_array():
    a_2D = [[1, 2, 3],
          [4, 5, 6]]

    a_1D = [-2, 0.5, 1.5]
    a2_2D = [[7, 8], [9, 10], [11, 12]]

    output3 = Array(a_1D) @ Array(a_1D)
    expected_output3 = np.dot(a_1D,a_1D)
    assert output3 == expected_output3

    
    output4 = Array(a_2D) @ Array(a_1D)
    expected_output4 = np.dot(a_2D,a_1D)
    assert output4.tolist() == expected_output4.tolist()

    output5 = Array(a_2D) @ Array(a2_2D)
    expected_output5 = np.dot(a_2D,a2_2D)
    assert output5.tolist() == expected_output5.tolist()