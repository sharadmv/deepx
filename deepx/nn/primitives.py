from ..core import Data

def Scalar(**kwargs):
    return Data((), datatype='Scalar', **kwargs)

def Vector(dim, **kwargs):
    return Data(dim, datatype='Vector', **kwargs)

def Matrix(dim1, dim2, **kwargs):
    return Data((dim1, dim2), datatype='Matrix', **kwargs)

def Image(dim1, dim2, dim3, **kwargs):
    return Data((dim1, dim2, dim3), datatype='Image', **kwargs)

def Sequence(data, max_length=None):
    return data.make_sequence(max_length)
