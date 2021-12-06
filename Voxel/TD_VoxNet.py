import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv3D, MaxPooling3D
from keras.metrics import categorical_accuracy
import keras.optimizers as opt
from keras.engine import data_adapter


class TVNModel(Sequential):
    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)
        return self.compute_metrics(x, y, y_pred, sample_weight)


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