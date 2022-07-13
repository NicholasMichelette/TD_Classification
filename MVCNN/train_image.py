import tensorflow as tf
import numpy as np
import os
import time
import random
import argparse
import matplotlib.pyplot as plt
from mvcnn_model import MVCNN
from keras.callbacks import CSVLogger

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Name of dataset to use to train.")
parser.add_argument("--epochs", nargs='?', default=8, type=int)
parser.add_argument("--save_epochs", nargs='?', default=10, type=int)
parser.add_argument("--batchsize", nargs='?', default=32, type=int)
parser.add_argument("--validation_split", nargs='?', default=0.15, type=float)
parser.add_argument("--image_width", nargs='?', default=227, type=int)
parser.add_argument("--image_height", nargs='?', default=227, type=int)
parser.add_argument("--neural_network", nargs='?', default=0, type=int, help='0 [default] for alexnet, 1 for simple classifier')
parser.add_argument("--vizualize", nargs='?', default=False, type=bool)
args = parser.parse_args()

def main():
    data_dir = os.path.join(os.getcwd(), "data", args.dataset, 'train')
    rseed = random.randint(0, 1000000)
    model_save_path = os.path.join("models", args.dataset)
    if os.path.exists(model_save_path) == False:
        os.mkdir(model_save_path)
    saved_models = os.listdir(model_save_path)
    base_name = str(args.image_height) + "x" + str(args.image_width) + "_"
    if args.neural_network==1:
        base_name = "simple_" + base_name
    save_name = base_name
    i = 0
    while save_name in saved_models:
        save_name = base_name + str(i)
        i += 1
    model_save_path = os.path.join(model_save_path, save_name)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = args.validation_split,
        subset = 'training',
        seed = rseed,
        image_size = (args.image_height, args.image_width),
        batch_size = args.batchsize)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = args.validation_split,
        subset = 'validation',
        seed = rseed,
        image_size = (args.image_height, args.image_width),
        batch_size = args.batchsize)
    
    AUTOTUNE = tf.data.AUTOTUNE
    classes = train_ds.class_names
    #train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    #val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    model_obj = MVCNN(len(classes), (args.image_height, args.image_width, 3))
    if args.neural_network == 1:
        model = model_obj.create_simple_classifier()
    else:
        model = model_obj.create_alexnet()
    model.summary()

    lpath = os.path.join('models', 'logs')
    if os.path.exists(lpath) == False:
        os.mkdir(lpath)
    sname = args.dataset + '_' + save_name
    logpath = os.path.join(lpath, sname + ".csv")
    log_time_file = os.path.join(lpath, sname + "_time" + ".txt")
    logger = CSVLogger(logpath, separator=",", append=True)
    with tf.device('/GPU:0'):
        t1 = time.time()
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[logger])
        t1 = t1 - time.time()
        f = open(log_time_file, 'w')
        f.write(str(t1))
        f.close()
        model.save(model_save_path)

    if args.vizualize:
        #print(history.history)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(args.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()


if __name__ == "__main__":
    main()