from setuptools import setup, find_packages

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
)
