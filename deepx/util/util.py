import theano
import theano.tensor as T

def create_tensor(ndim, name=None):
    if ndim == 1:
        return T.vector(name)
    elif ndim == 2:
        return T.matrix(name)
    elif ndim == 3:
        return T.tensor3(name)
    else:
        return T.TensorType(theano.config.floatX, (False,)*ndim)(name)

def shape_length(shape):
    if isinstance(shape, tuple):
        return sum(map(shape_length, shape))
    return 1

def pack_tuple(data, shape):
    def pack(data, shape):
        if shape is None:
            return None
        elif shape is 1:
            return data.pop(0)
        d = ()
        for s in shape:
            d += (pack(data, s),)
        return d
    return pack(list(data), shape)

def unpack_tuple(data):
    if isinstance(data, tuple):
        result  = ()
        shape = ()
        for da in data:
            d, s = unpack_tuple(da)
            if isinstance(d, tuple):
                result += d
            elif d is not None:
                result += (d,)
            shape += (s,)
        return result, shape
    else:
        if data is None:
            return data, None
        return data, 1
