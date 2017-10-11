"""
Used the backend initialization script from Keras. Thanks a lot to @fchollet for the hard work!
"""
import os
import json
import logging

BACKENDS = {
    'mxnet', 'tensorflow', 'theano'
}

_deepx_dir = os.path.expanduser(os.path.join('~', '.deepx'))
if not os.path.exists(_deepx_dir):
    os.makedirs(_deepx_dir)

DEFAULT_BACKEND = 'tensorflow'

_config_path = os.path.expanduser(os.path.join('~', '.deepx', 'deepx.json'))
if os.path.exists(_config_path):
    _config = json.load(open(_config_path))
    _floatx = _config.get('floatx', None)
    assert _floatx in {'float32', 'float64'}
    _epsilon = _config.get('epsilon', None)
    assert type(_epsilon) == float
    _backend = _config.get('backend', DEFAULT_BACKEND)
    assert _backend in BACKENDS

else:
    _floatx = 'float32'
    _epsilon = 1e-7
    _backend = DEFAULT_BACKEND

    _config = {
        'floatx': _floatx,
        'epsilon': _epsilon,
        'backend': _backend
    }

    with open(_config_path, 'w') as f:
        f.write(json.dumps(_config) + '\n')

if 'DEEPX_BACKEND' in os.environ:
    _backend = os.environ['DEEPX_BACKEND']
    assert _backend in BACKENDS

_BACKEND = _backend
try:
    if _BACKEND == 'theano':
        from .theano_backend import TheanoBackend as Backend
    elif _BACKEND == 'tensorflow':
        from .tensorflow_backend import TensorflowBackend as Backend
    elif _BACKEND == 'mxnet':
        from .mxnet_backend import MXNetBackend as Backend
    else:
        raise Exception('Unknown backend: ' + str(_BACKEND))
    backend = Backend()
    backend.set_floatx(_floatx)
    backend.set_epsilon(_epsilon)
    logging.info("Backend: %s", _BACKEND)
except:
    logging.exception("Failed importing: {backend}".format(backend=_BACKEND))
    raise Exception('Import failed: ' + str(_BACKEND))
