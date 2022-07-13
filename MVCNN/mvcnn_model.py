import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, Input, BatchNormalization, Rescaling
from keras.metrics import categorical_accuracy
import keras.optimizers as opt
from keras.engine import data_adapter


class TVNModel(Sequential):
    def test_step(self, data):
        #self.run_eagerly = True
        x, y = data
        test = x.numpy()
        y_pred_list = []
        for t in test:
            _y_pred = self(t, training=False)
            t_y_pred = _y_pred[0]
            for i in range(1, len(_y_pred)):
                t_y_pred = t_y_pred + _y_pred[i]
            t_y_pred = t_y_pred.numpy()
            t_y_pred = t_y_pred/len(_y_pred)
            y_pred_list.append(t_y_pred)

        y_pred_list = np.asarray(y_pred_list)

        #y_pred = self(x, training=False)
        y_pred = tf.convert_to_tensor(y_pred_list)
        
        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        #self.run_eagerly = False
        return {m.name: m.result() for m in self.metrics}


class MVCNN:
    def __init__(self, num_classes, input_shape, learning_rate=0.001):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.input_shape=input_shape

    def create_cnn1(self):
        model = Sequential()
        model.add(Rescaling(1./255, input_shape = self.input_shape))
        model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4,4), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
        model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))
        model.add(Flatten())
        
        self.model = model
        return model

    def create_alexnet(self):
        model = self.create_cnn1()
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        opti = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, clipnorm=1)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opti, metrics=['accuracy'])
        self.model = model
        return model

    def create_simple_classifier(self):
        model = Sequential()
        model.add(Rescaling(1./255, input_shape = self.input_shape))
        model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D())
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D())
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        opti = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, clipnorm=1)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opti, metrics=['accuracy'])
        self.model = model
        return model