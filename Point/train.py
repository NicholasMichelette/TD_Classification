import numpy as np
import os
import argparse
import tensorflow as tf
from TD_PointNet import TPointNet
from keras.callbacks import CSVLogger


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='Dataset to preprocess.')
parser.add_argument('--num_points_in', default=1024, type=int, help='Number of points to sample from the data.')
parser.add_argument('--num_points_out', default=1024, type=int, help='Number of points to sample from the data.')
parser.add_argument("--epochs", nargs='?', default=8, type=int)
parser.add_argument("--save_epochs", nargs='?', default=10, type=int)
parser.add_argument("--batchsize", nargs='?', default=32, type=int)
parser.add_argument("--validation_split", nargs='?', default=0.1, type=float)
args = parser.parse_args()


# shuffle code from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]


def main():
	datapath = os.path.join(os.getcwd(), "data", args.dataset)
	train_x_filename = os.path.join(datapath, args.dataset + "_" + str(args.num_points_in) + "_" + "train_x.npy")
	train_y_filename = os.path.join(datapath, args.dataset + "_" + str(args.num_points_in) + "_" + "train_y.npy")
	train_x = []
	train_y = []
	if os.path.exists(train_x_filename):
		train_x = np.load(train_x_filename)
	else:
		print("train_x.npy Data missing for dataset " + args.dataset + " with " + str(args.num_points_in) + "input points. Please run preprocess.py")
		quit()
	if os.path.exists(train_y_filename):
		train_y = np.load(train_y_filename)
	else:
		print("train_y.npy Data missing for dataset " + args.dataset + " with " + str(args.num_points_in) + "input points. Please run preprocess.py")
		quit()

	po = args.num_points_out
	if po > args.num_points_in:
		po = args.num_points_in
	TN = TPointNet(len(train_y[0]), train_x[0].shape, out_points = po)
	model = TN.create_model()
	train_x, train_y = unison_shuffled_copies(train_x, train_y)
	savepath = os.path.join(os.getcwd(), "models")
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	savename = args.dataset
	save_name = args.dataset + "_" + str(args.num_points_in) + "_" + str(po) + "_" + str(args.epochs) + "_"
	saved_models = os.listdir(savepath)
	while save_name in saved_models:
		save_name = save_name + str(1)
	logpath = os.path.join(os.getcwd(), "logs", save_name + ".csv")
	if not os.path.exists(os.path.join(os.getcwd(), "logs")):
		os.mkdir(os.path.join(os.getcwd(), "logs"))
	logger = CSVLogger(logpath, separator=",", append=True)
	with tf.device('/GPU:0'):
		model.fit(train_x, train_y, batch_size=args.batchsize, epochs=args.epochs, validation_split=args.validation_split, callbacks=[logger])
		model.save(os.path.join(savepath, save_name))
		


if __name__ == "__main__":
	main()
