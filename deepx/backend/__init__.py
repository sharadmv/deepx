"""
Used the backend initialization script from Keras. Thanks a lot to @fchollet for the hard work!
"""
import os
import json
import logging

CONFIG = {
    'backend': 'tensorflow',
    'epsilon': 1e-7,
    'floatx': 'float32'
}

deepx_dir = os.path.expanduser(os.path.join('~', '.deepx'))
if not os.path.exists(deepx_dir):
    os.makedirs(deepx_dir)

config_path = os.path.join(deepx_dir, 'deepx.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    CONFIG.update(config)
else:
    with open(config_path, 'w') as fp:
        json.dump(CONFIG, fp)

CONFIG['backend'] = os.environ.get("DEEPX_BACKEND", CONFIG["backend"])

backend_name = CONFIG["backend"]
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
