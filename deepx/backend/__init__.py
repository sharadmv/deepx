"""
Used the backend initialization script from Keras. Thanks a lot to @fchollet for the hard work!
"""
import os
import json
import logging
from deepx.config import CONFIG

backend_name = CONFIG['backend']

try:
    if backend_name == 'tensorflow':
        from deepx.backend.tensorflow import TensorflowBackend as Backend
    elif backend_name == 'theano':
        from deepx.backend.theano_backend import TheanoBackend as Backend
    elif backend_name == 'pytorch':
        from deepx.backend.pytorch import PyTorchBackend as Backend
    elif backend_name == 'jax':
        from deepx.backend.jax import JaxBackend as Backend
    else:
        raise Exception('Unknown backend: ' + str(backend_name))
    backend = Backend()
    backend.set_floatx(CONFIG["floatx"])
    backend.set_epsilon(CONFIG["epsilon"])
    logging.info("Backend: %s", backend_name)
except:
    logging.exception("Failed importing: {backend}".format(backend=backend_name))
    raise Exception('Import failed: {backend}'.format(backend=backend_name))

T = backend
