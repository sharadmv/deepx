from ..core import Data
from ..core import Shape

def Scalar(**kwargs):
    return Data(Shape(()), **kwargs)

def Vector(dim, name=None, **kwargs):
    assert isinstance(dim, int)
    return Data(Shape([dim], **kwargs), name=name)

def Matrix(a, b, name=None, **kwargs):
    return Data(Shape([a, b], **kwargs), name=name)

def Image(a, b, c, name=None, **kwargs):
    return Data(Shape([a, b, c], **kwargs), name=name)

def Sequence(data, max_length=None):
    shape = data.shape
    if data.sequence:
        raise Exception("data is already sequence")
    return data.copy(Shape(shape=[max_length] + shape.get_shape(), sequence=True))

def Value(value, dim, name=None, batch=None, **kwargs):
    return Data(Shape(dim, batch=batch), placeholder=value, name=name, **kwargs)

def Batch(data, batch_size=None):
    shape = data.shape
    if data.batch:
        raise Exception("data is already batch")
    return data.copy(Shape(shape=[batch_size] + shape.get_shape(), batch=True))
