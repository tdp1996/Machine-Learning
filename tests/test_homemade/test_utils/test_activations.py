from math import isclose
import numpy as np
from homemade.utils.activations import sigmoid
from homemade.utils.classess import Array

def test_activations():
    z = [1, 2, 3]
    output = sigmoid(z=Array(z))
    expected_output = 1 / (1 + np.exp(-(np.array(z))))
    assert output.tolist() == expected_output.tolist()