from math import isclose
import numpy as np
from utilities.activations import sigmoid

def test_activations():
    x = 0.1
    output_sigmoid = sigmoid(x=x)
    expected_output = 1 / (1 + np.exp(-x))
    assert isclose(output_sigmoid, expected_output)