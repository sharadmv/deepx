# Copyright 2019, mass=momentum_mass Google LLC
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

"""A basic MNIST example using JAX together with the mini-libraries stax, for
neural network building, and optimizers, for first-order stochastic optimization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import itertools
import tqdm

import numpy.random as npr

import jax.numpy as np
from jax import jit, grad
from jax.scipy.special import logsumexp
from jax.experimental import optimizers
from examples import datasets
import deepx.config
deepx.config.set_backend("jax")

from deepx import nn

def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -np.mean(preds * targets)

def accuracy(params, batch):
  inputs, targets = batch
  target_class = np.argmax(targets, axis=1)
  predicted_class = np.argmax(predict(params, inputs), axis=1)
  return np.mean(predicted_class == target_class)

network = (
        nn.Convolution([5, 5, 64]) >> nn.Relu() >> nn.Pool()
        >> nn.Convolution([5, 5, 64]) >> nn.Relu() >> nn.Pool()
        >> nn.Flatten() >> nn.Relu(1024) >> nn.Linear(10)
)
def predict(params, inputs):
    logits = network(inputs, params=params)
    logits = logits - logsumexp(logits, axis=1, keepdims=True)
    return logits

if __name__ == "__main__":
  step_size = 1e-3
  num_epochs = 10
  batch_size = 128

  train_images, train_labels, test_images, test_labels = datasets.mnist()
  train_images = train_images.reshape([-1, 28, 28, 1])
  test_images = test_images.reshape([-1, 28, 28, 1])
  network(train_images[:1])
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

  opt_init, opt_update = optimizers.adam(step_size)

  @jit
  def update(i, opt_state, batch):
    params = optimizers.get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  init_params = network.parameters
  opt_state = opt_init(init_params)
  itercount = itertools.count()

  print("\nStarting training...")
  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in tqdm.trange(num_batches):
      opt_state = update(next(itercount), opt_state, next(batches))
    epoch_time = time.time() - start_time

    params = optimizers.get_params(opt_state)
    test_acc = accuracy(params, (test_images, test_labels))
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Test set accuracy {}".format(test_acc))
