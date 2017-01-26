from setuptools import setup, find_packages

required = []
recommended = {
    'tensorflow': ['tensorflow==0.12.0'],
    'theano': ['Theano==0.9.0b1'],
    'mxnet': [],
}

setup(
    name = "deepx",
    version = "0.2.0",
    author = "Sharad Vikram and Zachary Chase Lipton",
    author_email = "sharad.vikram@gmail.com",
    description = "A basic deep learning library.",
    license = "MIT",
    keywords = "theano, tensorflow",
    packages=find_packages(
        '.'
    ),
    classifiers=[
    ],
    extra_requires = recommended,
)
