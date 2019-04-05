import numpy as np
import tensorflow as tf
import tqdm
from tensorflow import keras

from deepx import keras as layers
from deepx import nn
from deepx.backend import T

T.set_default_device(T.gpu())

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

img_rows, img_cols = 28, 28
num_classes = 10

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

batch_size = 128

x = T.placeholder(T.floatx(), [None, 28, 28, 1])
y = T.placeholder(T.int32, [None, 10])

network = (
        nn.Convolution([5, 5, 64]) >> nn.Relu() >> nn.Pool()
        >> nn.Convolution([5, 5, 64]) >> nn.Relu() >> nn.Pool()
        >> nn.Flatten() >> nn.Relu(200) >> nn.Linear(10)
)
logits = network(x)
cost = T.mean(T.categorical_crossentropy(output=logits, target=y, from_logits=True))
accuracy = T.mean(T.cast(T.equal(T.argmax(y, axis=-1),
                          T.argmax(logits, axis=-1)),
                  T.floatx()))
optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)
hm_epochs = 10

if __name__ == "__main__":
    with T.session(allow_soft_placement=True, allow_growth=True) as sess:

        N = x_train.shape[0]
        for epoch in range(hm_epochs):
            permutation = np.random.permutation(N)
            for i in tqdm.trange(N // batch_size + 1):
                batch_idx = permutation[i * batch_size:(i + 1) * batch_size]
                _, c = sess.run([optimizer, cost], feed_dict={
                    x: x_train[batch_idx],
                    y: y_train[batch_idx]
                })
            print("Accuracy:", sess.run(accuracy, {
                x: x_test,
                y: y_test
            }))
