import numpy as np
import os
import argparse
from TD_PointNet import TPointNet


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='Dataset to preprocess.')
parser.add_argument('--num_points_in', default=1024, type=int, help='Number of points to sample from the data.')
args = parser.parse_args()


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

	print(train_x[0].shape)
	TN = TPointNet(len(train_y), train_x[0].shape)
	model = TN.create_model()


if __name__ == "__main__":
	main()
