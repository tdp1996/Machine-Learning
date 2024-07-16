import math
from typing import Union
from .array import Array

def sigmoid(z: Union[Array,int,float]):
    return 1 / (1 + Array.exp(-z))