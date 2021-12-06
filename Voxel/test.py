import argparse
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model


parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Name of dataset to use to test on. Must have already run voxelize_raw_data.py.")
parser.add_argument("model", help="Name of the saved model in Voxel\Models directory.")
parser.add_argument("rotations", nargs='?', help="How many rotations an object has. Can be found at the end of file names in data\classname\test",
                    default=12, type=int)
args = parser.parse_args()


def main():
    datapath = os.getcwd() + "\\data\\" + args.dataset
    if os.path.isdir(datapath):
        test_x = []
        test_y = []
        classes = os.listdir(datapath)
        num_classes = len(classes)
        for i in range(num_classes):
            cdp = datapath + "\\" + classes[i] + "\\test"
            class_matrix = np.zeros(10)
            class_matrix[i] = 1
            files = os.listdir(cdp)
            for file in files:
                fp = cdp + "\\" + file
                test_x = test_x + list(np.load(fp))
                for j in range(int(args.rotations)):
                    test_y.append(class_matrix)

        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)
        test_x = np.expand_dims(test_x, axis=(4))
        model = load_model(os.getcwd() + "\\models\\" + args.model)
        with tf.device('/GPU:0'):
            model.evaluate(test_x, test_y, batch_size=32)
            


if __name__ == "__main__":
    main()