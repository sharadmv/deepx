# DeepX [![Build Status](https://travis-ci.org/sharadmv/deepx.svg?branch=master)](https://travis-ci.org/sharadmv/deepx) [![Coverage Status](https://coveralls.io/repos/sharadmv/deepx/badge.svg?branch=master&service=github)](https://coveralls.io/github/sharadmv/deepx?branch=master) [![PyPI](https://img.shields.io/pypi/v/deepx.svg)](https://pypi.python.org/pypi/deepx)
DeepX is a deep learning library designed with flexibility and succinctness in mind.
The key aspect is an expressive shorthand to describe your neural network architecture.

DeepX supports both [Theano](http://deeplearning.net/software/theano/) and [Tensorflow](http://www.tensorflow.org).

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
>>> from deepx.nn import *
>>> mlp = Vector(784) >> Tanh(200) >> Tanh(200) >> Softmax(10)
```
Another way of writing the same net would be:
```python
>>> mlp = Vector(784) >> Repeat(Tanh(200), 2) >> Softmax(10)
```

Our model has a `predict` method, which allows us to pass data through the network. Let's test it with
some dummy data:
```python

>>> mlp(np.ones((10, 784)))
```

10 is our batch size in this example.

Sweet! We now have an model that can predict MNIST classes! To start learning the parameters
of our model, we first want to define a loss function. Let's use cross entropy loss.

```python
>>> from deepx.loss import *
>>> loss = mlp >> CrossEntropy()
```

Finally, we want to set up an optimization algorithm to minimize loss. An optimization algorithm takes in
a model and a loss function.

```python
>>> from deepx.optimize import *
>>> rmsprop = RMSProp(loss)
```

Finally, to perform gradient descent updates, we just call the `train` method of `rmsprop`.

```python
>>> rmsprop.train(X_batch, y_batch, learning_rate)
```

That's it, we're done!
