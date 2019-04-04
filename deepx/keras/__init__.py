from abc import ABCMeta
from functools import wraps
import inspect
import tensorflow.keras as keras

from deepx.keras.wrapper import KerasLayer

attrs = [
    (name, getattr(keras.layers, name)) for name in dir(keras.layers)
]
layers = [(name, a) for (name, a) in attrs if inspect.isclass(a) and issubclass(a, keras.layers.Layer)]

for name, layer in layers:
    def foo(layer):
        @wraps(layer)
        def layer_func(*args, **kwargs):
            return KerasLayer(layer(*args, **kwargs))
        return layer_func
    globals()[name] = foo(layer)

__all__ = [name for (name, _) in layers]
