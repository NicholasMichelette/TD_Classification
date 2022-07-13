import argparse
import os
import random
import numpy as np
import tensorflow as tf
import PIL
from matplotlib import image
from matplotlib import pyplot
from keras.metrics import categorical_accuracy
from keras.models import load_model


parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Name of dataset to use to test on.")
parser.add_argument("model", help="Name of the saved model in Voxel\Models directory.")
parser.add_argument("--batchsize", nargs='?', default=32, type=int)
parser.add_argument("--image_width", nargs='?', default=227, type=int)
parser.add_argument("--image_height", nargs='?', default=227, type=int)
parser.add_argument("--predict_path", nargs='?', default="")
args = parser.parse_args()

def main():
    data_dir = os.path.join(os.getcwd(), "data", args.dataset, 'test')
    classes = os.listdir(data_dir)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size = (args.image_height, args.image_width))

    model = load_model(os.path.join("models", args.dataset, args.model))

    with tf.device('/GPU:0'):
        if args.predict_path != "":
            class_path = os.path.join(data_dir, classes[random.randrange(len(classes))])
            imgs = os.listdir(class_path)
            img_path = os.path.join(class_path, imgs[random.randrange(len(imgs))])
            image = PIL.Image.open(img_path).resize((args.image_width, args.image_height))
            img = np.asarray(image)
            img_np_expanded = np.expand_dims(img, axis=0)
            pred = model.predict(img_np_expanded)
            top_k_values, top_k_indicies = tf.nn.top_k(pred, k=3)
            pred_ind = top_k_indicies.numpy().tolist()[0]
            pred_value = top_k_values.numpy().tolist()[0]
            print(classes[pred_ind[0]])
            pyplot.title(classes[pred_ind[0]])
            pyplot.imshow(image)
            pyplot.show()
        else:
            model.evaluate(test_ds, batch_size=args.batchsize)

if __name__=='__main__':
    main()