import cv2
import argparse
import os
import random
import numpy as np
import tensorflow as tf
from keras.metrics import categorical_accuracy
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Name of dataset to use to test on.")
parser.add_argument("model", help="Name of the saved model in Voxel\Models directory.")
parser.add_argument("--image_width", nargs='?', default=227, type=int)
parser.add_argument("--image_height", nargs='?', default=227, type=int)
parser.add_argument("--gray", nargs='?', default=False, type=bool)
args = parser.parse_args()


def main():
    cap = cv2.VideoCapture(4)
    model = load_model(os.path.join("models", args.dataset, args.model))
    labels = os.listdir(os.path.join("data", args.dataset, "train"))

    with tf.device('/GPU:0'):
        while True:
                ret, frame_orig = cap.read()  # reads the frame
                root_wind = 'Classifier'
                cv2.namedWindow(root_wind)
                
                frame = cv2.resize(frame_orig, (args.image_width, args.image_height))
                if args.gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                frame_np_expanded = np.expand_dims(frame, axis=0)
                pred = model.predict(frame_np_expanded)
                top_k_values, top_k_indicies = tf.nn.top_k(pred, k=3)
                pred_ind = top_k_indicies.numpy().tolist()[0]
                pred_value = top_k_values.numpy().tolist()[0]
                print(labels[pred_ind[0]])

                cv2.imshow(root_wind, frame_orig)

                if cv2.waitKey(20) == ord('q'):  # Introduce 20 milisecond delay. press q to exit.
                    break

if __name__=='__main__':
    main()