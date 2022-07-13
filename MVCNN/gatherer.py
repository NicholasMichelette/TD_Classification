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
parser.add_argument("--classname", default="")
parser.add_argument("--image_width", nargs='?', default=227, type=int)
parser.add_argument("--image_height", nargs='?', default=227, type=int)
parser.add_argument("--gray", nargs='?', default=False, type=bool)
args = parser.parse_args()


def main():
    cap = cv2.VideoCapture(4)
    data_path = os.path.join('data', args.dataset, 'train')
    if os.path.exists(data_path) == False:
        os.mkdir(data_path)
    classes = os.listdir(data_path)
    next_class_name = len(classes)
    if args.classname != "":
        next_class_name = args.classname
    class_path = os.path.join(data_path, str(next_class_name))
    if os.path.exists(class_path) == False:
        os.mkdir(class_path)
    c = 0

    with tf.device('/GPU:0'):
        while True:
                ret, frame_orig = cap.read()  # reads the frame
                root_wind = 'Classifier'
                cv2.namedWindow(root_wind)
                
                frame = cv2.resize(frame_orig, (args.image_width, args.image_height))
                if args.gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                cv2.imwrite(os.path.join(class_path, str(c) + '.jpg'), frame)
                c += 1
                #pred = model.predict(frame_np_expanded)
                #top_k_values, top_k_indicies = tf.nn.top_k(pred, k=3)
                #pred_ind = top_k_indicies.numpy().tolist()[0]
                #pred_value = top_k_values.numpy().tolist()[0]
                #print(labels[pred_ind[0]])

                cv2.imshow(root_wind, frame_orig)

                if cv2.waitKey(20) == ord('q'):  # Introduce 20 milisecond delay. press q to exit.
                    break

if __name__=='__main__':
    main()