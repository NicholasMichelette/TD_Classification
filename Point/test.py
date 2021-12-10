import numpy as np
import os
import argparse
import tensorflow as tf
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Name of dataset to use to test on. Must have already run preprocess.py(should have been run when training).")
parser.add_argument("model", help="Name of the saved model in Point\models directory.")
args = parser.parse_args()


def main():
	datapath = os.path.join(os.getcwd(), "data", args.dataset)
	num_points_in = args.model.split("_")[1]
	test_x_filename = os.path.join(datapath, args.dataset + "_" + str(num_points_in) + "_" + "test_x.npy")
	test_y_filename = os.path.join(datapath, args.dataset + "_" + str(num_points_in) + "_" + "test_y.npy")
	test_x = []
	test_y = []
	if os.path.exists(test_x_filename):
		test_x = np.load(test_x_filename)
	else:
		print("test_x.npy Data missing for dataset " + args.dataset + " with " + str(args.num_points_in) + "input points. Please run preprocess.py")
		quit()
	if os.path.exists(test_y_filename):
		test_y = np.load(test_y_filename)
	else:
		print("test_y.npy Data missing for dataset " + args.dataset + " with " + str(args.num_points_in) + "input points. Please run preprocess.py")
		quit()

	model = load_model(os.path.join(os.getcwd(), "models", args.model))
	with tf.device('/GPU:0'):
		model.evaluate(test_x, test_y, batch_size=32)

if __name__ == "__main__":
	main()
