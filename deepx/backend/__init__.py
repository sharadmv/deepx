"""
Ported the backend library from Keras. Thanks a lot to @fchollet for the hard work!
"""
import os
import json
import logging
from .common import epsilon, floatx, set_epsilon, set_floatx

_deepx_dir = os.path.expanduser(os.path.join('~', '.deepx'))
if not os.path.exists(_deepx_dir):
    os.makedirs(_deepx_dir)

_BACKEND = 'theano'
_config_path = os.path.expanduser(os.path.join('~', '.deepx', 'deepx.json'))
if os.path.exists(_config_path):
    _config = json.load(open(_config_path))
    _floatx = _config.get('floatx', floatx())
    assert _floatx in {'float32', 'float64'}
    _epsilon = _config.get('epsilon', epsilon())
    assert type(_epsilon) == float
    _backend = _config.get('backend', _BACKEND)
    assert _backend in {'theano', 'tensorflow'}

    set_floatx(_floatx)
    set_epsilon(_epsilon)
    _BACKEND = _backend
else:
    # save config file, for easy edition
    _config = {'floatx': floatx(),
               'epsilon': epsilon(),
               'backend': _BACKEND}
    with open(_config_path, 'w') as f:
        # add new line in order for bash 'cat' display the content correctly
        f.write(json.dumps(_config) + '\n')

if 'DEEPX_BACKEND' in os.environ:
    _backend = os.environ['DEEPX_BACKEND']
    assert _backend in {'theano', 'tensorflow'}
    _BACKEND = _backend

if _BACKEND == 'theano':
    from .theano_backend import *
elif _BACKEND == 'tensorflow':
    from .tensorflow_backend import *
else:
    raise Exception('Unknown backend: ' + str(_BACKEND))
logging.info("Backend: %s", _BACKEND)
