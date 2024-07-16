import math
from typing import Union
from .classess import Array

def sigmoid(z: Union[Array,int,float]):
    return 1 / (1 + Array.exp(-z))