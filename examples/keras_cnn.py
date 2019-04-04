from tensorflow import keras
from deepx import keras as layers
from deepx import nn

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
net = (
    nn.Repeat(layers.Conv2D(32, (3, 3)) >> layers.ReLU(), 2)
    >> layers.Conv2D(64, (3, 3)) >> layers.ReLU()
    >> layers.MaxPooling2D(pool_size=(2, 2))
    >> layers.Dropout(0.25)
    >> layers.Flatten()
    >> layers.Dense(128) >> layers.ReLU()
    >> layers.Dropout(0.5)
    >> layers.Dense(num_classes) >> layers.Softmax()
)
outputs = net(inputs)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
