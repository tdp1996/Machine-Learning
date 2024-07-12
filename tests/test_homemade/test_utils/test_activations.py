from math import isclose
import numpy as np
from homemade.utils.activations import sigmoid

def test_activations():
    z = 0.1
    output_sigmoid = sigmoid(z=z)
    expected_output = 1 / (1 + np.exp(-z))
    assert isclose(output_sigmoid, expected_output)