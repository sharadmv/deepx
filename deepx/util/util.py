import theano.tensor as T

def create_tensor(ndim, name=None):
    if ndim == 1:
        return T.vector(name)
    elif ndim == 2:
        return T.matrix(name)
    elif ndim == 3:
        return T.tensor3(name)
