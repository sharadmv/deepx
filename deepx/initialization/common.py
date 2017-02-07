from deepx import T

def initialize_weights(name, shape, value=None, **kwargs):
    return globals()[name](shape, **kwargs)

def glorot_uniform(shape):
    return T.random_uniform(shape)
    print(shape)
    pass
