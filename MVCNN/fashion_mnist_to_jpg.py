import keras
from keras.datasets import fashion_mnist

import numpy as np
from PIL import Image, ImageOps
import os

class_names = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

def save_image(filename, data_array):
    im = Image.fromarray(data_array.astype('uint8'))
    im_invert = ImageOps.invert(im)
    im_invert.save(filename)

# Load Fashion-MNIST Data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

DIR_NAME = "mnist_fashon"
train_dir = os.path.join(DIR_NAME, 'train')
test_dir = os.path.join(DIR_NAME, 'test')
if os.path.exists(DIR_NAME) == False:
    os.mkdir(DIR_NAME)
    os.mkdir(train_dir)
    os.mkdir(test_dir)

# Save Images
i = 0

#u = 0
#for x in x_train:
#    savedir = os.path.join(train_dir, str(y_train[u]))
#    if os.path.exists(savedir) == False:
#        os.mkdir(savedir)
#    filename = "{0}/{1:05d}.jpg".format(savedir,i)
#    save_image(filename, x)
#    i += 1
#    u += 1


u = 0
for x in x_test:
    savedir = os.path.join(test_dir, str(class_names[y_test[u]]))
    if os.path.exists(savedir) == False:
        os.mkdir(savedir)
    filename = "{0}/{1:05d}.jpg".format(savedir,i)
    save_image(filename, x)
    i += 1
    u += 1