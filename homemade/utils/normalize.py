"""Normalize features"""

from typing import Union
from .analysis import calculate_mean, calculate_standard_deviation
from .array import Array


def normalize(features: Array):
    features_normalized = features.copy()

    features_mean = calculate_mean(features, axis=0)

    features_deviation = calculate_standard_deviation(features, axis=0)

    features_deviation.data = [
        deviation if deviation > 0 else 1 for deviation in features_deviation.data
    ]

    features_normalized -= features_mean

    features_normalized /= features_deviation

    return features_normalized
