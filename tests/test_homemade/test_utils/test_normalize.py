import numpy as np
from sklearn.preprocessing import StandardScaler
from homemade.utils.array import Array
from homemade.utils.normalize import normalize

def test_normalize():
    data = [[1, 2, 3, 4, 5],[10, 20, 30, 40, 50]]
    np_array = np.array(data)
    normalized_data = normalize(Array(data))
    scaler = StandardScaler()
    expected_output = scaler.fit_transform(np_array)
    assert normalized_data.tolist() == expected_output.tolist()