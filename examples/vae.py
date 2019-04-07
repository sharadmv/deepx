import numpy as np
import tqdm
import deepx.config
backend = deepx.config.get_backend()
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
if backend == 'pytorch':
    import torch

from deepx.backend import T
from deepx import nn, stats
from deepx.stats import layers

(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()
mnist_images = np.random.binomial(1, mnist_images / 255.)
mnist_images = tf.cast(tf.reshape(mnist_images, [-1, 784]), tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((
    mnist_images,
    tf.cast(mnist_labels,tf.int64)))
mnist_train = dataset.shuffle(1000).batch(128)

D = 784
L = 40

p_Z = stats.GaussianDiag(T.zeros(L), T.ones(L))

decoder = (
    nn.Relu(L, 500) >> nn.Relu(500) >> layers.Bernoulli(D)
)
encoder = (
    nn.Relu(D, 500) >> nn.Relu(500) >> layers.Gaussian(L)
)
train_vars = decoder.get_parameters() + encoder.get_parameters()


def loss(images):
    q_Z = encoder(images)
    q_X = decoder(q_Z.sample())
    kl = T.mean(stats.kl_divergence(q_Z, p_Z))
    log_likelihood = T.mean(T.sum(q_X.log_likelihood(images), axis=-1))
    elbo = log_likelihood - kl
    return -elbo, log_likelihood, kl

if backend == 'tensorflow':
    optimizer = tf.train.AdamOptimizer(1e-3)
    for _ in range(10):
        for (batch, (images, _)) in enumerate(tqdm.tqdm(mnist_train, total=469)):
            with tf.GradientTape() as tape:
                nelbo, _, _ = loss(images)
                grads = tape.gradient(nelbo, train_vars)
            optimizer.apply_gradients(zip(grads, train_vars),
                                    global_step=tf.train.get_or_create_global_step())
        print(tuple(a.numpy() for a in loss(mnist_images[:1000])))
elif backend == 'pytorch':
    optimizer = torch.optim.Adam(train_vars, 1e-3)
    for _ in range(10):
        for (batch, (images, _)) in enumerate(tqdm.tqdm(mnist_train, total=469)):
            images = torch.tensor(images.numpy()).to(T.get_default_device())
            optimizer.zero_grad()
            nelbo, _, _ = loss(images)
            nelbo.backward()
            optimizer.step()
        images = torch.tensor(mnist_images[:1000].numpy()).to(T.get_default_device())
        print(tuple(np.asscalar(a.detach().cpu().numpy()) for a in loss(images)))
