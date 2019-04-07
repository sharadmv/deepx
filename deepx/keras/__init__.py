from abc import ABCMeta
from functools import wraps
import inspect
import tensorflow.keras as keras

from deepx.backend import T
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

def Input(shape=None,
          batch_size=None,
          name=None,
          dtype=None,
          sparse=False,
          tensor=None,
          **kwargs):
    if 'batch_shape' in kwargs:
        batch_shape = kwargs.pop('batch_shape')
        if shape and batch_shape:
            raise ValueError('Only provide the shape OR '
                            'batch_shape argument to '
                            'Input, not both at the same time.')
        batch_size = batch_shape[0]
        shape = batch_shape[1:]
    if kwargs:
        raise ValueError('Unrecognized keyword arguments:', kwargs.keys())

    if dtype is None:
        dtype = T.floatx()
    if shape is None and tensor is None:
        raise ValueError('Please provide to Input either a `shape`'
                        ' or a `tensor` argument. Note that '
                        '`shape` does not include the batch '
                        'dimension.')
    input_layer = InputLayer(
        input_shape=shape,
        batch_size=batch_size,
        name=name,
        dtype=dtype,
        sparse=sparse,
        input_tensor=tensor)
    # Return tensor including `_keras_history`.
    # Note that in this case train_output and test_output are the same pointer.
    outputs = input_layer._inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs

__all__ = [name for (name, _) in layers] + ["Input"]
