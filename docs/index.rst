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
This shorthand is
an abstract definition of
a neural network that can plug in
to several choices of backend.
Specifically, we support 
[Tensorflow](http://www.tensorflow.org), [PyTorch](https://pytorch.org/)
and [Jax](https://github.com/google/jax).

DeepX is agnostic to the framework
you use for training and deployment, and only aims
to make defining complex neural network
architectures simple. However, it is very modular
and fits into any training/deployment pipeline seamlessly.

Quickstart
------------------------------
Let's consider the task of building
a multilayer perceptron (MLP) in DeepX
that will classify MNIST digits
We first define
the structure of the network.

.. code-block:: python

  from deepx import nn
  network = nn.Relu(200) >> nn.Relu(200) >> nn.Softmax(10) 

Note that we did not tell the network the input size (784,
in the case of MNIST). DeepX is *lazy*
and won’t initialize weights in
the network until the last minute. In this case,
DeepX won’t know the input size until we actually pass
something in.

Let's try something a bit harder: a convolutional neural network classifier

.. code-block:: python

    from deepx import nn
    network = (
        nn.Reshape([28, 28, 1])
        >> nn.Convolution([5, 5, 64]) >> nn.Relu() >> nn.Pool()
        >> nn.Convolution([5, 5, 64]) >> nn.Relu() >> nn.Pool()
        >> nn.Flatten() >> nn.Relu(200) >> nn.Relu(200) >> nn.Softmax(10)
    )

This network definition is a bit repetitive, so DeepX offers
higher order functions that can help definitions be a bit more concise.

.. code-block:: python

    from deepx import nn
    network = (
        nn.Reshape([28, 28, 1])
        >> nn.Repeat(nn.Convolution([5, 5, 64]) >> nn.Relu() >> nn.Pool(), 2)
        >> nn.Flatten() >> nn.Repeat(nn.Relu(200), 2) >> nn.Softmax(10)
    )

:code:`nn.Repeat` is syntactic sugar, so we get the exact same network 
in both the previous definitinos.

Example usage
------------------------------
In all cases, these neural networks are *functions*, that take in batches
of vectors as inputs
and return output vectors. How you use them is specific to the
backend you are using and its paradigm. For example,
if we are using Tensorflow (graph), this network
would used for graph construction, but if we are using Tensorflow (eager) or PyTorch,
the network would be called in the training loop.



Tensorflow (graph):

.. code-block:: python
    
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    prediction = network(x)
    weights = network.get_parameters()
    loss = loss_function(label, prediction)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, var_list=weights)
    with tf.Session() as sess:
        for i in range(num_iters):
            image_, label_ = next_batch()
            sess.run(train_op, {
              image: image_,
              label: label_
            })

Tensorflow (eager):

.. code-block:: python

    optimizer = tf.train.AdamOptimizer(1e-3)
    weights = network.get_parameters()
    for i in range(num_iters):
        with tf.GradientTape() as tape:
            image, label = next_batch()
            prediction = network(image)
            loss = loss_function(label, prediction)
            grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))

Pytorch:

.. code-block:: python

    weights = network.get_parameters()
    optimizer = torch.optim.Adam(weights, 1e-3)
    for i in range(num_iters):
        image, label = next_batch()
        optimizer.zero_grad()
        prediction = network(image)
        loss = loss_function(label, prediction)
        loss.backward()
        optimizer.step()


.. automodule:: deepx.nn

Table of Contents
=======================

.. toctree::
   :caption: API

   user/tutorial
   source/modules
