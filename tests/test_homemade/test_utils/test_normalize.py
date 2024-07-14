# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from homemade.utils.normalize import normalize

# def test_normalize():
#     data = [[1, 2, 3, 4, 5],[10, 20, 30, 40, 50]]
#     np_array = np.array(data)
#     normalized_data = normalize(data)
#     scaler = StandardScaler()
#     expected_output = scaler.fit_transform(np_array)
#     assert np.allclose(normalized_data,expected_output,atol=1e-8)