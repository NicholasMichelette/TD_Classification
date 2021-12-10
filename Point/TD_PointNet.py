import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, Conv2D, MaxPooling1D
from keras.metrics import categorical_accuracy
import keras.optimizers as opt
from keras.engine import data_adapter

class TPointNet:
    def __init__(self, num_classes, input_shape, learning_rate=0.001, out_points=1024):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.input_shape=input_shape
        self.out_points = out_points

    def create_model(self):
        model = Sequential()
        model.add(Conv1D(64, 1, activation="relu", input_shape=self.input_shape))
        model.add(Conv1D(64, 1, activation="relu"))
        model.add(Conv1D(64, 1, activation="relu"))
        model.add(Conv1D(128, 1, activation="relu"))
        model.add(Conv1D(self.out_points, 1, activation="relu"))
        model.add(MaxPooling1D(pool_size=self.out_points))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation="softmax"))
        opti = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, clipnorm=1)
        model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=[categorical_accuracy])
        #print(model.summary())
        #print(model.input_shape)
        self.model = model
        return model
