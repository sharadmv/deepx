from setuptools import setup, find_packages

import deepx

recommended = {
    'tensorflow': ['tensorflow'],
    'pytorch': ['torch'],
    'jax': ['jax', 'jaxlib'],
}

setup(
    name = "deepx",
    version = deepx.__version__,
    author = "Sharad Vikram",
    author_email = "sharad.vikram@gmail.com",
    description = "A basic deep learning library.",
    license = "MIT",
    keywords = "theano, tensorflow, pytorch",
    packages=['deepx'],
    classifiers=[],
    extras_require = recommended,
)
