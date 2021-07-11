from keras.models import Sequential

from keras.layers import LayerNormalization
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.layers.normalization import BatchNormalization

def make_model(classes, input_shape):
    model = Sequential()

    model.add(Conv2D(96, (7, 7), padding = "same", input_shape = input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(LayerNormalization())

    model.add(Conv2D(256, (5, 5), padding = "same"))
    model.add(Activation("relu"))  
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(LayerNormalization())

    model.add(Conv2D(384, (3, 3), padding = "same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(LayerNormalization())

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation("relu"))

    model.add(Dense(512))
    model.add(Activation("relu"))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    model.summary()

    return model
