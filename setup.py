from setuptools import setup, find_packages

setup(
    name = "deepx",
    version = "0.0.1",
    author = "Sharad Vikram and Zachary Chase Lipton",
    author_email = "sharad.vikram@gmail.com",
    license = "MIT",
    keywords = "theano",
    install_requires=['theanify==0.0.19'],
    packages=find_packages(),
    classifiers=[
    ],
)
