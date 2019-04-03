from setuptools import setup, find_packages

import deepx

recommended = {
    'tensorflow': ['tensorflow'],
    'theano': ['Theano==1.0.1'],
    'mxnet': [],
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
    classifiers=[
    ],
    extra_requires = recommended,
)
