from setuptools import setup, find_packages

setup(
    name = "deepx",
    version = "0.1.0",
    author = "Sharad Vikram and Zachary Chase Lipton",
    author_email = "sharad.vikram@gmail.com",
    description = "A basic deep learning library for sequence learning built using Theano.",
    license = "MIT",
    keywords = "theano",
    install_requires = ['theano'],
    packages=find_packages(
        '.'
    ),
    classifiers=[
    ],
)
