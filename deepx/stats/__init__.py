import deepx.config as _config

_backend = _config.get_backend()
if _backend == 'tensorflow':
    from deepx.stats.tensorflow import *
elif _backend == 'pytorch':
    from deepx.stats.pytorch import *
elif _backend == 'jax':
    from deepx.stats.jax import *
