import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.metrics import categorical_accuracy
import keras.optimizers as opt
from keras.engine import data_adapter

class TPointNet:
    def __init__(self, num_classes, input_shape, learning_rate=0.001, ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.input_shape=input_shape

    def create_model(self):
        model = Sequential()
        model.add(Dense(64, activation="relu", input_shape=self.input_shape))
        model.add(Dense(64, activation="relu"))
        opti = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, clipnorm=1)
        model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=[categorical_accuracy])
        print(model.summary())
        self.model = model
        return model
