import numpy as np

from ..core import Data
from ..core import Shape

def Scalar(**kwargs):
    return Data(Shape(()), datatype='Scalar', **kwargs)

def Vector(dim, name=None, **kwargs):
    assert isinstance(dim, int)
    return Data(Shape(dim, **kwargs), datatype='Vector', name=name)

def Matrix(dim, name=None, **kwargs):
    assert len(dim) == 2
    return Data(Shape(dim, **kwargs), datatype='Matrix', name=name)

def Image(dim, name=None, **kwargs):
    assert len(dim) == 3
    return Data(Shape(dim, **kwargs), datatype='Image', name=name)

def Sequence(data, max_length=None):
    return data.make_sequence(max_length)
