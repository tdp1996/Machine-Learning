import numpy as np
import pytest
from homemade.utils.array import Array


def test_add_Array():
    a_2D = [[1, 2, 3], [4, 5, 6]]
    a_1D = [-1, 0.5, 2]
    scalar = 3

    # test add scalar to array 2D
    output1 = Array(a_2D) + scalar
    expected_output1 = np.add(a_2D, scalar)
    assert output1.tolist() == expected_output1.tolist()

    # test add scalar to array 1D
    output2 = Array(a_1D) + scalar
    expected_output2 = np.add(a_1D, scalar)
    assert output2.tolist() == expected_output2.tolist()

    # test add array 1D to array 1D
    output3 = Array(a_1D) + Array(a_1D)
    expected_output3 = np.add(a_1D, a_1D)
    assert output3.tolist() == expected_output3.tolist()

    # test add an array 2D to an array 2D
    output4 = Array(a_2D) + Array(a_2D)
    expected_output4 = np.add(a_2D, a_2D)
    assert output4.tolist() == expected_output4.tolist()

    # test add an array 1D to an array 2D
    output5 = Array(a_1D) + Array(a_2D)
    expected_output5 = np.add(a_1D, a_2D)
    assert output5.tolist() == expected_output5.tolist()

def test_add_array_error():
    with pytest.raises(
        ValueError, match=r"operands could not be broadcast together with shapes"
    ):
        a_2D_incompatible = [[1, 2], [3, 4]]
        a_1D_incompatible = [1, 2, 3]
        Array(a_2D_incompatible) + Array(a_1D_incompatible)

    with pytest.raises(
        ValueError, match=r"operands could not be broadcast together with shapes"
    ):
        a_2D = [[1, 2, 3], [4, 5, 6]]
        a_2D_incompatible = [[1, 2], [3, 4]]
        Array(a_2D) + Array(a_2D_incompatible)

def test_sub_array():
    a_2D = [[1, 2, 3], [4, 5, 6]]
    a_1D = [-1, 0.5, 2]
    scalar = 3

    # test subtract scalar from array 2D
    output1 = Array(a_2D) - scalar
    expected_output1 = np.subtract(a_2D, scalar)
    assert output1.tolist() == expected_output1.tolist()

    # test add scalar from array 1D
    output2 = Array(a_1D) - scalar
    expected_output2 = np.subtract(a_1D, scalar)
    assert output2.tolist() == expected_output2.tolist()

    # test subtract array 1D from array 1D
    output3 = Array(a_1D) - Array(a_1D)
    expected_output3 = np.subtract(a_1D, a_1D)
    assert output3.tolist() == expected_output3.tolist()

    # test subtract an array 2D from an array 2D
    output4 = Array(a_2D) - Array(a_2D)
    expected_output4 = np.subtract(a_2D, a_2D)
    assert output4.tolist() == expected_output4.tolist()

    # test subtract an array 1D from array 2D
    output5 = Array(a_2D) - Array(a_1D)
    expected_output5 = np.subtract(a_2D, a_1D)
    assert output5.tolist() == expected_output5.tolist()

def test_radd_array():
    a_2D = [[1, 2, 3], [4, 5, 6]]
    a_1D = [-1, 0.5, 2]
    scalar = 3

    output1 = scalar + Array(a_2D) 
    expected_output1 = np.add(scalar, a_2D)
    assert output1.tolist() == expected_output1.tolist()

    output2 = scalar + Array(a_1D) 
    expected_output2 = np.add(scalar, a_1D)
    assert output2.tolist() == expected_output2.tolist()

def test_subtract_array_error():
    with pytest.raises(
        ValueError, match=r"operands could not be broadcast together with shapes"
    ):
        a_2D_incompatible = [[1, 2], [3, 4]]
        a_1D_incompatible = [1, 2, 3]
        Array(a_2D_incompatible) - Array(a_1D_incompatible)

    with pytest.raises(
        ValueError, match=r"operands could not be broadcast together with shapes"
    ):
        a_2D = [[1, 2, 3], [4, 5, 6]]
        a_2D_incompatible = [[1, 2], [3, 4]]
        Array(a_2D) - Array(a_2D_incompatible)

def test_rsub_array():
    a_2D = [[1, 2, 3], [4, 5, 6]]
    a_1D = [-1, 0.5, 2]
    scalar = 3

    output1 = scalar - Array(a_2D) 
    expected_output1 = np.subtract(scalar, a_2D)
    assert output1.tolist() == expected_output1.tolist()

    output2 = scalar - Array(a_1D) 
    expected_output2 = np.subtract(scalar, a_1D)
    assert output2.tolist() == expected_output2.tolist()

def test_multiply_array():
    a_2D = [[1, 2, 3], [4, 5, 6]]
    a_1D = [-1, 0.5, 2]
    scalar = 3

    # test multiply an array 2D with a scalar
    output1 = Array(a_2D) * scalar
    expected_output1 = np.multiply(a_2D, scalar)
    assert output1.tolist() == expected_output1.tolist()

    # test multiply an array 1D with a scalar
    output2 = Array(a_1D) * scalar
    expected_output2 = np.multiply(a_1D, scalar)
    assert output2.tolist() == expected_output2.tolist()

    # test multiply two arrays 1D
    output3 = Array(a_1D) * Array(a_1D)
    expected_output3 = np.multiply(a_1D, a_1D)
    assert output3.tolist() == expected_output3.tolist()

    # test multiply two arrays 2D
    output4 = Array(a_2D) * Array(a_2D)
    expected_output4 = np.multiply(a_2D, a_2D)
    assert output4.tolist() == expected_output4.tolist()

    # test multiply an array 2D with an array 1D
    output5 = Array(a_2D) * Array(a_1D)
    expected_output5 = np.multiply(a_2D, a_1D)
    assert output5.tolist() == expected_output5.tolist()

def test_multiply_array_error():
    with pytest.raises(
        ValueError, match=r"operands could not be broadcast together with shapes"
    ):
        a_2D_incompatible = [[1, 2], [3, 4]]
        a_1D_incompatible = [1, 2, 3]
        Array(a_2D_incompatible) * Array(a_1D_incompatible)

    with pytest.raises(
        ValueError, match=r"operands could not be broadcast together with shapes"
    ):
        a_2D = [[1, 2, 3], [4, 5, 6]]
        a_2D_incompatible = [[1, 2], [3, 4]]
        Array(a_2D) * Array(a_2D_incompatible)

def test_rmul_array():
    a_2D = [[1, 2, 3], [4, 5, 6]]
    a_1D = [-1, 0.5, 2]
    scalar = 3

    output1 = scalar * Array(a_2D) 
    expected_output1 = np.multiply(scalar, a_2D)
    assert output1.tolist() == expected_output1.tolist()

    output2 = scalar * Array(a_1D) 
    expected_output2 = np.multiply(scalar, a_1D)
    assert output2.tolist() == expected_output2.tolist()

def test_truediv_array():
    a_2D = [[1, 2, 3], [4, 5, 6]]
    a_1D = [-1, 0.5, 2]
    scalar = 3

    # test divide an array 2D with a scalar
    output1 = Array(a_2D) / scalar
    expected_output1 = np.divide(a_2D, scalar)
    assert output1.tolist() == expected_output1.tolist()

    # test divide an array 1D with a scalar
    output2 = Array(a_1D) / scalar
    expected_output2 = np.divide(a_1D, scalar)
    assert output2.tolist() == expected_output2.tolist()

    # test divide two arrays 1D
    output3 = Array(a_1D) / Array(a_1D)
    expected_output3 = np.divide(a_1D, a_1D)
    assert output3.tolist() == expected_output3.tolist()

    # test divide two arrays 2D
    output4 = Array(a_2D) / Array(a_2D)
    expected_output4 = np.divide(a_2D, a_2D)
    assert output4.tolist() == expected_output4.tolist()

    # test divide an array 2D with an array 1D
    output5 = Array(a_2D) / Array(a_1D)
    expected_output5 = np.divide(a_2D, a_1D)
    assert output5.tolist() == expected_output5.tolist()

def test_truediv_array_error():
    with pytest.raises(
        ValueError, match=r"operands could not be broadcast together with shapes"
    ):
        a_2D_incompatible = [[1, 2], [3, 4]]
        a_1D_incompatible = [1, 2, 3]
        Array(a_2D_incompatible) / Array(a_1D_incompatible)

    with pytest.raises(
        ValueError, match=r"operands could not be broadcast together with shapes"
    ):
        a_2D = [[1, 2, 3], [4, 5, 6]]
        a_2D_incompatible = [[1, 2], [3, 4]]
        Array(a_2D) / Array(a_2D_incompatible)

def test_rtruediv_array():
    a_2D = [[1, 2, 3], [4, 5, 6]]
    a_1D = [-1, 0.5, 2]
    scalar = 3

    output1 = scalar / Array(a_2D) 
    expected_output1 = np.divide(scalar, a_2D)
    assert output1.tolist() == expected_output1.tolist()

    output2 = scalar / Array(a_1D) 
    expected_output2 = np.divide(scalar, a_1D)
    assert output2.tolist() == expected_output2.tolist()

def test_dot_array():
    a_2D = [[1, 2, 3], [4, 5, 6]]

    a_1D = [-2, 0.5, 1.5]
    a2_2D = [[7, 8], [9, 10], [11, 12]]

    # test dot two array 1D
    output1 = Array(a_1D) @ Array(a_1D)
    expected_output1 = np.dot(a_1D, a_1D)
    assert output1 == expected_output1

    # test dot an array 2D with array 1D
    output2 = Array(a_2D) @ Array(a_1D)
    expected_output2 = np.dot(a_2D, a_1D)
    assert output2.tolist() == expected_output2.tolist()

    # test dot two array 2D
    output3 = Array(a_2D) @ Array(a2_2D)
    expected_output3 = np.dot(a_2D, a2_2D)
    assert output3.tolist() == expected_output3.tolist()

def test_dot_array_error():
    with pytest.raises(ValueError, match=r"Dot product not supported for these shapes"):
        a_2D_incompatible = [[1, 2], [3, 4]]
        a_1D_incompatible = [1, 2, 3]
        Array(a_2D_incompatible) @ Array(a_1D_incompatible)

    with pytest.raises(ValueError, match=r"Dot product not supported for these shapes"):
        a_2D = [[1, 2, 3], [4, 5, 6]]
        a_2D_incompatible = [[1, 2], [3, 4]]
        Array(a_2D) @ Array(a_2D_incompatible)

def test_sum_array():
    a_2D = [[1, 2, 3], [4, 5, 6]]
    a_1D = [-2, 0.5, 1.5]

    output1 = Array.sum(a_1D)
    expected_output1 = np.sum(a_1D)
    assert output1 == expected_output1

    output2 = Array.sum(a_2D)
    expected_output2 = np.sum(a_2D)
    assert output2 == expected_output2

    output3 = Array.sum(a_2D, axis=0)
    expected_output3 = np.sum(a_2D, axis=0)
    assert output3.tolist() == expected_output3.tolist()

    output4 = Array.sum(a_2D, axis=1)
    expected_output4 = np.sum(a_2D, axis=1)
    assert output4.tolist() == expected_output4.tolist()

    output5 = Array.sum(a_1D, axis=0)
    expected_output5 = np.sum(a_1D, axis=0)
    assert output5 == expected_output5

def test_sum_array_error():
    with pytest.raises(ValueError, match=r"Axis 1 is not valid for array with shape"):
        a_incompatible = [1, 2, 3]
        Array.sum(a_incompatible, axis=1)
    with pytest.raises(ValueError, match=r"Invalid axis"):
        a2D_incompatible = [[1, 2, 3], [4, 5, 6]]
        Array.sum(a2D_incompatible, axis=2)

def test_pow_array():
    a_2D = [[1, 2, 3], [4, 5, 6]]
    a_1D = [-2, 0.5, 1.5]

    output1 = Array(a_1D).__pow__(3)
    expected_output1 = np.pow(a_1D, 3)
    assert output1.tolist() == expected_output1.tolist()

    output2 = Array(a_2D).__pow__(3)
    expected_output2 = np.pow(a_2D, 3)
    assert output2.tolist() == expected_output2.tolist()

def test_sqrt_array():
    a_2D = [[1, 2, 3], [4, 5, 6]]
    a_1D = [1, 0.5, 1.5]

    output1 = Array.sqrt(a_1D)
    expected_output1 = np.sqrt(a_1D)
    assert output1.tolist() == expected_output1.tolist()

    output2 = Array.sqrt(a_2D)
    expected_output2 = np.sqrt(a_2D)
    assert output2.tolist() == expected_output2.tolist()

def test_sqrt_array_error():
    with pytest.raises(ValueError, match=r"Cannot compute square root of negative number"):
        a_incompatible = [1, -2, 3]
        Array.sqrt(a_incompatible)
    with pytest.raises(ValueError, match=r"Cannot compute square root of negative number"):
        a2D_incompatible = [[1, 2, -3], [4, 5, 6]]
        Array.sqrt(a2D_incompatible)

def test_log_array():
    a_2D = [[1, 2, 3], [4, 5, 6]]
    a_1D = [1, 0.5, 1.5]

    output1 = Array.log(a_1D)
    expected_output1 = np.log(a_1D)
    assert output1.tolist() == expected_output1.tolist()

    output2 = Array.log(a_2D)
    expected_output2 = np.log(a_2D)
    assert output2.tolist() == expected_output2.tolist()

def test_log_array_error():
    with pytest.raises(ValueError, match=r"log method only supports positive values."):
        a_incompatible = [1, -2, 3]
        Array.log(a_incompatible)
    with pytest.raises(ValueError, match=r"log method only supports positive values."):
        a2D_incompatible = [[1, 2, -3], [4, 5, 6]]
        Array.log(a2D_incompatible)

def test_abs_array():
    a_2D = [[1, -2, 3], [4, 5, -6]]
    a_1D = [1, -0.5, -1.5]

    output1 = Array.abs(a_1D)
    expected_output1 = np.abs(a_1D)
    assert output1.tolist() == expected_output1.tolist()

    output2 = Array.abs(a_2D)
    expected_output2 = np.abs(a_2D)
    assert output2.tolist() == expected_output2.tolist()

def test_exp_array():
    a_2D = [[1, -2, 3], [4, 5, -6]]
    a_1D = [1, -0.5, -1.5]

    output1 = Array.exp(a_1D)
    expected_output1 = np.exp(a_1D)
    assert output1.tolist() == expected_output1.tolist()

    output2 = Array.exp(a_2D)
    expected_output2 = np.exp(a_2D)
    assert output2.tolist() == expected_output2.tolist()

