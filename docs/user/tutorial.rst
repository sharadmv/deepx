.. _tutorial:

Tutorial
=================
The main element of DeepX's shorthand
is the chain operator, `>>`.
The chain operator takes two
networks
and connects the output of
one into the inputs of the other.

For example, take
two example networks `A` and `B`.
Chain acts as a composition
operator, so `A >> B` would be
a new neural network, whose input
first passes through `A` and then
through `B`.

In DeepX, everything can be
seen as a "network",
so building a complicated
or deeper networks is just a
series of compositions of
smaller networks.

Example Networks (`deepx.nn`)
-------------------------------
DeepX is declarative,
so it's probably easiest
to learn by example.


#. A multilayer perceptron that classifies MNIST.

    .. code-block:: python

        mlp = Vector(784) >> Tanh(200) >> Tanh(200) >> Softmax(10)


    DeepX provides functions that can help compress
    your neural network description, while keeping it as readable.
    An equivalent network is:

    .. code-block:: python

        mlp = Vector(784) >> Repeat(Tanh(200), 2) >> Softmax(10)

#. A convolutional neural network that classifies MNIST.

    .. code-block:: python

        convnet = Image((1, 28, 28)) >> Conv((10, 2, 2)) >> Conv((20, 2, 2)) >> Flatten() >> Softmax(10)

    In DeepX, the `Conv` node is just shorthand
    for a `Convolution` followed by an activation function (`Relu`),
    and then a `Pool`.

    .. code-block:: python

        >>> Image((1, 28, 28)) >> Conv((10, 2, 2))
        Image((1, 28, 28)) >> Convolution((1, 28, 28), (10, 28, 28)) >> Relu() >> Pool((10, 28, 28), (10, 14, 14))


#. A network that simply concatenates two vectors.

    .. code-block:: python

        network = (Vector(5), Vector(5)) >> Concatenate()

    An alternate way of writing this using operator overloading is

    .. code-block:: python

        network = Vector(5) | Vector(5)

Minimizing loss functions (`deepx.loss`, `deepx.optimize`)
-------------------------------------------------------------

After creating a network definition,
we generally aim to minimize some
loss function over a dataset.

In classification, a common loss function to use
is cross entropy.

.. code-block:: python

    mlp = Vector(784) >> Repeat(Tanh(200), 2) >> Softmax(10)
    loss = mlp >> CrossEntropy()

Loss functions are treated as nodes in a network
that output scalar values. The one difference
between a loss function and a normal
neural network layer is that a loss function
typically accepts two inputs: the network output
and a set of targets.

In DeepX by default, adding
a loss function
will implicitly add an input
for the targets.
However, you can also explictly pass
in a target node.

.. code-block:: python
    
    input = Vector(784)
    output = input >> Repeat(Tanh(200), 2) >> Softmax(10)
    target = Vector(10)
    loss = (output, target) >> CrossEntropy()

Note that the chain operator allows
multiple inputs to a node.

Finally, after we have a network
that produces a loss,
we can optimize.

.. code-block:: python

    loss = Vector(784) >> Repeat(Tanh(200), 2) >> Softmax(10) >> CrossEntropy()
    optimizer = SGD(loss)
    optimizer.train(X_train, y_train, learning_rate)
