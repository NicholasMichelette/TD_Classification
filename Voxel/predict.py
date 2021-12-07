import argparse
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model


parser = argparse.ArgumentParser()
parser.add_argument("model", help="Name of the saved model in Voxel\Models directory.")
parser.add_argument("datafile", help="Full filepath of file to predict from.")
args = parser.parse_args()


def predict_from_model(model, object):
    return model.predict_step(object)


def main():
    model = load_model(os.path.join(os.getcwd(), "models", args.model))
    object = np.load(os.path.load(os.getcwd(), args.datafile))
    object = np.expand_dims(object, axis=(4))
    predict_list = []
    with tf.device('/GPU:0'):
        for o in object:
            predict_list.append(predict_from_model(model, np.expand_dims(o, axis=(0))))

    prediction = np.zeros(len(predict_list[0]))
    for p in predict_list:
        prediction = prediction + p[0]
    prediction = prediction.numpy()
    prediction = prediction/len(predict_list)
    thresh = prediction < 0.001
    prediction[thresh] = 0
    print(prediction)


if __name__ == "__main__":
    main()