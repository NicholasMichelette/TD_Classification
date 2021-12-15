import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv3D, MaxPooling3D
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


class TVoxNet:
    def __init__(self, num_classes, input_shape, learning_rate=0.001, ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.input_shape=input_shape

    def create_model(self):
        model = Sequential()
        model.add(Conv3D(32, (5,5,5), strides=(2,2,2), activation="relu", input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(Conv3D(32, (3,3,3), activation="relu"))
        model.add(MaxPooling3D(pool_size=(2,2,2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.num_classes, activation="softmax"))
        opti = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, clipnorm=1)
        model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=[categorical_accuracy])
        self.model = model
        return model


    def create_tvn_model(self):
        model = TVNModel()
        model.add(Conv3D(32, (5,5,5), strides=(2,2,2), activation="relu", input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(Conv3D(32, (3,3,3), activation="relu"))
        model.add(MaxPooling3D(pool_size=(2,2,2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.num_classes, activation="softmax"))
        opti = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, clipnorm=1)
        model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=[categorical_accuracy])
        self.model = model
        print(model.summary())
        return model