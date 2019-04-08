.. _tutorial:

Tutorial
=================
The main element of DeepX's shorthand
is the compose operator, :code:`>>`.
The compose operator takes two
networks
and connects the output of
one into the inputs of the other.

For example, take
two example networks :code:`A` and :code:`B`.
:code:`A >> B` would be
a new neural network, whose input
first passes through :code:`A` and then
through :code:`B`.

In DeepX, everything can be
seen as a "network",
so building a complicated
or deeper networks is just a
series of compositions of
smaller networks.

Example Networks
--------------------------------------
DeepX is declarative,
so it's probably easiest
to learn by example.

For starters, you can import layers
by running :code:`from deepx import nn`.

#. A multilayer perceptron that classifies MNIST.

    .. code-block:: python

        network = nn.Relu(784, 200) >> nn.Relu(200) >> nn.Softmax(10)


   DeepX provides functions that can help compress
   your neural network description, while keeping it as readable.
   An equivalent network is:

    .. code-block:: python

        network = nn.Repeat(nn.Relu(200), 2) >> nn.Softmax(10)

#. A convolutional neural network that classifies MNIST.

    .. code-block:: python

        network = (
            nn.Reshape([28, 28, 1])
            >> nn.Repeat(nn.Convolution([5, 5, 64]) >> nn.Relu() >> nn.Pool(), 2)
            >> nn.Flatten() >> nn.Repeat(nn.Relu(200), 2) >> nn.Softmax(10)
        )

    In DeepX, :code:`Conv` is syntactic
    for a :code:`Convolution` followed by an activation function (by default :code:`Relu`),
    and then a :code:`Pool`, so we can further compress this definition.

    .. code-block:: python

        network = (
            nn.Reshape([28, 28, 1])
            >> nn.Repeat(nn.Conv([5, 5, 64]), 2)
            >> nn.Flatten() >> nn.Repeat(nn.Relu(200), 2) >> nn.Softmax(10)
        )

#. A network with dropout.

    .. code-block:: python

        network = nn.Tanh(200) >> nn.Dropout(0.5) >> nn.Tanh(200) >> nn.Dropout(0.5) >> nn.Softmax(10)

    or equivalently:

    .. code-block:: python

        network = nn.Repeat(nn.Tanh(200) >> nn.Dropout(0.5), 2) >> nn.Softmax(10)

To get details about the layers offered by DeepX,
please refer to the API docs.

Working with backends
-------------------------------------

DeepX is backend agnostic, but you need
to set the backend globally before
creating the network, so it knows
which library's functions to call.
Currently, DeepX supports
Tensorflow (both graph and eager),
Pytorch, and Jax. 

To choose a backend, 
you have several options, each which takes
priority over the next.

#. You can use :code:`deepx.config` to set the 
   backend before actually using any other 
   parts of DeepX. Specifically, you can do

   .. code-block:: python

         import deepx.config
         deepx.config.set_backend("<tensorflow|pytorch|jax>")


#. You can set set the environment variable
   :code:`DEEPX_BACKEND` to :code:`tensorflow`
   :code:`pytorch`, or :code:`jax`


#. DeepX generates a config file :code:`~/.deepx/deepx.json`,
   which you can edit.

Multiple GPU support
-----------------------

.. _Tensorflow: https://www.tensorflow.org/guide/using_gpu

DeepX borrows the `Tensorflow`_ style of
selecting devices to store network outputs,
specifically using :code:`with` statements.

.. code-block:: python

    with T.device(T.cpu()):
        y_cpu = network(x)

    with T.device(T.gpu(0)):
        y_gpu = network(x)

:code:`y_cpu` lives on CPU
and :code:`y_gpu` lives on GPU 0.
This syntax works for Tensorflow and Pytorch
but isn't yet supported for Jax.
