from deepx.initialization.common import *

def get_initializer(name):
    if not isinstance(name, str):
        return name
    if name not in globals():
        raise Exception("Initializer not found")
    return globals()[name]
