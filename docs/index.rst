DeepX: a minimalist library for deep learning
================================================

Welcome!
------------------------------
DeepX is a deep learning library for Python
that aims to make defining
and training a neural network
as simple as humanly possible.

Our approach is an intuitive shorthand
that allows concise, but expressive
model definitions.
This shorthand creates
a Tensorflow or Theano graph,
enabling efficient GPU
utilization.

Quickstart
------------------------------
Let's consider the task of building
a multilayer perceptron (MLP)
that will classify MNIST digits
in DeepX. We first define
the structure of the network.

.. code-block:: python

    from deepx.nn import Vector, Tanh, Softmax
    mlp = Vector(784) >> Tanh(200) >> Tanh(200) >> Softmax(10) 

In order to train this neural network,
we need to specify a loss function.

.. code-block:: python

    from deepx.loss import CrossEntropy
    loss = mlp >> CrossEntropy()

Finally, we need to use an
optimization algorithm to
minimize the loss function
with respect to minibatches of data.

.. code-block:: python

    from deepx.optimizer import Adam
    adam = Adam(loss)
    adam.train(x_batch, y_batch, learning_rate)

Check out the :doc:`tutorial <user/tutorial>` if you want to learn more!

Otherwise, you can browse the various
layers, loss functions, and
optimizers we have implemented so far.

Table of Contents
=======================

.. toctree::
    user/tutorial
