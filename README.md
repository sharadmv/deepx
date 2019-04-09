# DeepX [![Build Status](https://travis-ci.org/sharadmv/deepx.svg?branch=master)](https://travis-ci.org/sharadmv/deepx) [![Coverage Status](https://coveralls.io/repos/sharadmv/deepx/badge.svg?branch=master&service=github)](https://coveralls.io/github/sharadmv/deepx?branch=master) [![PyPI](https://img.shields.io/pypi/v/deepx.svg)](https://pypi.python.org/pypi/deepx)
DeepX is a deep learning library designed with flexibility and succinctness in mind.
The key aspect is an expressive shorthand to describe your neural network architecture.

DeepX supports [Tensorflow](http://www.tensorflow.org), [PyTorch](https://pytorch.org/)
and [Jax](https://github.com/google/jax). DeepX is also still in alpha, so there are likely
going to be changes that change the API/names/design. I'm very open to feedback and suggestions,
so shoot me an email if you are interested in contributing.

Installation
====================================

```
$ pip install deepx
```

Quickstart
=================================

The first step in building your first network is to define your *model*.
The model is the input-output structure of your network.
Let's consider the task of classifying MNIST with a multilayer perceptron (MLP).

```python
>>> from deepx import nn
>>> net = nn.Relu(200) >> nn.Relu(200) >> nn.Softmax(10)
```

Our model behaves like a function.
```python
import tensorflow as tf
net(tf.ones((10, 784)))
```
To get the weights out of the network, we can just say:
```python
net.get_parameters()
```

We can also use a convolutional neural network for classification and it'll work exactly the same!
```python
net = (nn.Reshape([28, 28, 1])
        >> nn.Conv([2, 2, 64])
        >> nn.Conv([2, 2, 32])
        >> nn.Conv([2, 2, 16])
        >> nn.Flatten() >> nn.Relu(200) >> nn.Relu(200) >> nn.Softmax(10))
```


That's it, we're done!

Keras
=====================================
DeepX allows you to use Keras layers with the same `>>` composition format.
It's really easy! All layers in `tf.keras.layers` are wrapped in the `deepx.keras` package.
This allows you to compose them. 

```python
import deepx.keras as nn
net = (
    nn.Conv2D(64, (5, 5), padding='same') >> nn.ReLU() >> nn.MaxPooling2D(padding='same')
    >> nn.Conv2D(64, (5, 5), padding='same') >> nn.ReLU() >> nn.MaxPooling2D(padding='same')
    >> nn.Flatten() >> nn.Dense(1024)
    >> nn.ReLU() >> nn.Dense(10) >> nn.Softmax()
)
```
These layers are only compatible with Tensorflow, however, since the only compatible backend between
Keras and DeepX is Tensorflow.

Distributions
========================
DeepX also wraps the distributions that ship with [Tensorflow Probability](https://www.tensorflow.org/probability) and [PyTorch](https://pytorch.org/docs/stable/distributions.html).
This enables you write probabilistic neural networks (like those in the VAE) very easily.
```python
decoder = nn.Relu(L, 500) >> nn.Relu(500) >> layers.Bernoulli(D)
encoder = nn.Relu(D, 500) >> nn.Relu(500) >> layers.Gaussian(L)
```
