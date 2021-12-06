import argparse
import os
import numpy as np
import tensorflow as tf
from TD_VoxNet import TVoxNet


parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Name of dataset to use to train. Must have already run voxelize_raw_data.py.")
parser.add_argument("--rotations", nargs='?', help="How many rotations an object has. Can be found at the end of file names in data\classname\train",
                    default=12, type=int)
parser.add_argument("--epochs", nargs='?', default=8, type=int)
parser.add_argument("--batchsize", nargs='?', default=32, type=int)
parser.add_argument("--validation_split", nargs='?', default=0.1, type=float)
args = parser.parse_args()


# shuffle code from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def main():
    datapath = os.getcwd() + "\\data\\" + args.dataset
    if os.path.isdir(datapath):
        train_x = []
        train_y = []
        classes = os.listdir(datapath)
        num_classes = len(classes)
        for i in range(num_classes):
            cdp = datapath + "\\" + classes[i] + "\\train"
            class_matrix = np.zeros(10)
            class_matrix[i] = 1
            files = os.listdir(cdp)
            for file in files:
                fp = cdp + "\\" + file
                train_x = train_x + list(np.load(fp))
                for j in range(int(args.rotations)):
                    train_y.append(class_matrix)
    
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        train_x = np.expand_dims(train_x, axis=(4))
        train_x, train_y = unison_shuffled_copies(train_x, train_y)
        vn = TVoxNet(num_classes, train_x.shape[1:])
        model = vn.create_model()
        #model.summary()
        with tf.device('/GPU:0'):
            model.fit(train_x, train_y, batch_size=args.batchsize, validation_split=args.validation_split, epochs=args.epochs)
            model.save(os.getcwd() + "\\models\\" + args.dataset + "_" + str(args.rotations) + "_" + str(args.epochs))

if __name__ == "__main__":
    main()