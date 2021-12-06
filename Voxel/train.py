import argparse
import os
import gc
import numpy as np
import tensorflow as tf
from TD_VoxNet import TVoxNet, TVNModel
from keras.metrics import categorical_accuracy
from keras.callbacks import CSVLogger


parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Name of dataset to use to train. Must have already run voxelize_raw_data.py.")
parser.add_argument("--rotations", nargs='?', help="How many rotations an object has. Can be found at the end of file names in data\classname\train",
                    default=12, type=int)
parser.add_argument("--epochs", nargs='?', default=8, type=int)
parser.add_argument("--save_epochs", nargs='?', default=10, type=int)
parser.add_argument("--batchsize", nargs='?', default=32, type=int)
parser.add_argument("--validation_split", nargs='?', default=0.1, type=float)
parser.add_argument("--manual_validation", nargs='?', default=True, type=bool)
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
        val_x = []
        val_y = []
        classes = os.listdir(datapath)
        num_classes = len(classes)
        for i in range(num_classes):
            cdp = datapath + "\\" + classes[i] + "\\train"
            class_matrix = np.zeros(10)
            class_matrix[i] = 1
            files = os.listdir(cdp)
            if args.manual_validation:
                fileslen = len(files)
                train_set = int(fileslen*(1 - args.validation_split))
                for fileindex in range(train_set):
                    file = files[fileindex]
                    fp = cdp + "\\" + file
                    train_x = train_x + list(np.load(fp))
                    for j in range(int(args.rotations)):
                        train_y.append(class_matrix)
                for fileindex in range(train_set, fileslen):
                    file = files[fileindex]
                    fp = cdp + "\\" + file
                    val_x.append(np.expand_dims(np.load(fp), axis=(4)))
                    val_y.append(class_matrix)
            else:
                for file in files:
                    fp = cdp + "\\" + file
                    train_x = train_x + list(np.load(fp))
                    for j in range(int(args.rotations)):
                        train_y.append(class_matrix)
    
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        train_x = np.expand_dims(train_x, axis=(4))
        train_x, train_y = unison_shuffled_copies(train_x, train_y)
        val_x = np.asarray(val_x)
        val_y = np.asarray(val_y)
        vn = TVoxNet(num_classes, train_x.shape[1:])
        if args.manual_validation:
            model = vn.create_tvn_model()
        else:
            model = vn.create_model()
        with tf.device('/GPU:0'):
            save_name = args.dataset + "_" + str(args.rotations) + "_" + str(args.epochs) + "_"
            saved_models = os.listdir(os.getcwd() + "\\models\\")
            while save_name in saved_models:
                save_name = save_name + str(1)
            logpath = os.path.join(os.getcwd(), "logs", save_name + ".csv")
            logger = CSVLogger(logpath, separator=",", append=True)
            if args.manual_validation:
                f = open(os.path.join(os.getcwd(), "logs", "val_" + save_name + ".csv"), 'w')
                f.write("epoch,categorical_accuracy,loss\n")
                for i in range(args.epochs):
                    epoch_str = str(i + 1) + "/" + str(args.epochs)
                    print("Epoch " + epoch_str + ":")
                    model.run_eagerly=False
                    model.fit(train_x, train_y, batch_size=args.batchsize, epochs=1, callbacks=[logger])
                    model.run_eagerly=True
                    eval = model.evaluate(val_x, val_y, batch_size=args.batchsize)
                    f.write(str(i + 1) + "," + str(eval[1]) + "," + str(eval[0]) + "\n")
                    if i % args.save_epochs == 0 and i != 0:
                        model.save(os.getcwd() + "\\models\\" + save_name)
                    gc.collect()
                    tf.keras.backend.clear_session()
                f.close()
            else:
                model.fit(train_x, train_y, batch_size=args.batchsize, validation_split=args.validation_split, epochs=args.epochs, callbacks=[logger])
            model.save(os.getcwd() + "\\models\\" + save_name)

if __name__ == "__main__":
    main()