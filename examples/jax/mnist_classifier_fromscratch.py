# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic MNIST example using Numpy and JAX.

The primary aim here is simplicity and minimal dependencies.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tqdm

import numpy.random as npr

import deepx.config
deepx.config.set_backend("jax")
from deepx import nn
from deepx.backend import T
from jax.api import jit, grad
from jax.config import config
from jax.scipy.special import logsumexp
import jax.numpy as np
from examples import datasets


def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
    return [(scale * rng.randn(m, n), scale * rng.randn(n))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

network = (
        nn.Relu(784, 200) >> nn.Relu(200) >> nn.Linear(10)
)
def predict(params, inputs):
    logits = network(inputs, params=params)
    return logits - logsumexp(logits, axis=1, keepdims=True)

def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -np.mean(preds * targets)

def accuracy(params, batch):
    inputs, targets = batch
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)

if __name__ == "__main__":
    layer_sizes = [784, 1024, 1024, 10]
    param_scale = 0.1
    step_size = 0.001
    num_epochs = 10
    batch_size = 128

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]
    batches = data_stream()

    def nested_update(params, grads):
        if isinstance(params, tuple):
            return tuple(nested_update(ps, gs) for
                        ps, gs in zip(params, grads))
        elif isinstance(params, list):
            return list(nested_update(ps, gs) for
                        ps, gs in zip(params, grads))
        elif isinstance(params, dict):
            return {k:nested_update(params[k], grads[k]) for
                    k in params.keys()}
        return params - step_size * grads

    @jit
    def update(params, batch):
        grads = grad(loss)(params, batch)
        return nested_update(params, grads)

    params = network.parameters
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in tqdm.trange(num_batches):
            params = update(params, next(batches))
        epoch_time = time.time() - start_time

        # train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        #print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))
