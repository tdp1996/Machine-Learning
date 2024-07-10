"""Normalize features"""

from typing import Union
from .analysis import calculate_mean
from .analysis import calculate_standard_deviation

def normalize(features: list[list[Union[float,int]]]):
    features_normalized = features.copy()

    features_mean = calculate_mean(features,axis=0)

    features_deviation = calculate_standard_deviation(features, axis=0)

    features_deviation = [deviation if deviation>0 else 1 for deviation in features_deviation]

    features_normalized = [[(features[i][j] - features_mean[j])/features_deviation[j] for j in range(len(features_mean))] 
                           for i in range(len(features))]

    return features_normalized