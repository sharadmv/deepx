import numpy as np

from ..core import Data
from ..core import Shape

def Scalar(**kwargs):
    return Data(Shape(()), datatype='Scalar', **kwargs)

def Vector(dim, **kwargs):
    assert isinstance(dim, int)
    return Data(Shape(dim, **kwargs), datatype='Vector')

def Matrix(dim, **kwargs):
    assert len(dim) == 2
    return Data(Shape(dim, **kwargs), datatype='Matrix')

def Image(dim, **kwargs):
    assert len(dim) == 3
    return Data(Shape(dim, **kwargs), datatype='Image')

def Sequence(data, max_length=None):
    return data.make_sequence(max_length)
