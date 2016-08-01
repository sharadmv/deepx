from .. import T

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
    if isinstance(data, tuple) or isinstance(data, list):
        result  = ()
        shape = ()
        for da in data:
            d, s = unpack_tuple(da)
            if isinstance(d, tuple) or isinstance(d, list):
                result += d
            elif d is not None:
                result += (d,)
            shape += (s,)
        return result, shape
    else:
        if data is None:
            return data, None
        return data, 1

def is_iterable(d):
    try:
        _ = (e for e in d)
        return True
    except:
        return False

def rzip(l1, l2):
    for a, b in zip(l1, l2):
        if is_iterable(a) and is_iterable(b):
            yield type(a)(rzip(a, b))
        else:
            yield a, b

def tile_data(from_data, to_data):
    dim = 0
    while T.ndim(from_data) < T.ndim(to_data):
        from_data = T.expand_dims(from_data, 0)
        to_dim = T.shape(to_data)[dim]
        from_data = T.tile(from_data, [to_dim] + [1] * (T.ndim(from_data) - 1))
        dim += 1
    return from_data

def concatenate_data(tensor_list, sequence=False):
    if not sequence:
        return T.concatenate(tensor_list, axis=-1)
    max_tensors = sorted([t for t in tensor_list], key=lambda x: T.ndim(x))[::-1]
    raw_tensors = max_tensors[0:1] + [tile_data(t, max_tensors[0]) for t in max_tensors[1:]]
    return T.concatenate(raw_tensors, axis=-1)

def flatten(tuples):
    return (t2 for t1 in tuples for t2 in t1)
