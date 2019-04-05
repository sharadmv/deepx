import tensorflow as tf
from tensorflow import keras
from deepx import keras as nn
from deepx.backend import T

T.set_default_device(T.gpu())
# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

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

inputs = keras.Input((28, 28, 1))
network = (
        nn.Conv2D(64, (5, 5), padding='same') >> nn.ReLU() >> nn.MaxPooling2D(padding='same')
        >> nn.Conv2D(64, (5, 5), padding='same') >> nn.ReLU() >> nn.MaxPooling2D(padding='same')
        >> nn.Flatten() >> nn.Dense(1024)
        >> nn.ReLU() >> nn.Dense(10) >> nn.Softmax()
)
outputs = network(inputs)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
keras.backend.set_session(sess)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss=T.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10)
test_scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
