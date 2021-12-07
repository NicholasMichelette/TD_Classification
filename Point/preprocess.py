import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import pointreader as pr
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='Dataset to preprocess.')
parser.add_argument('--num_points_in', default=1024, type=int, help='Number of points to sample from the data.')
args = parser.parse_args()


def main():
	datapath = os.path.join(os.path.dirname(os.getcwd()), "data_raw", args.dataset)
	savepath = os.path.join(os.getcwd(), "data", args.dataset)
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	classes = os.listdir(datapath)
	num_classes = len(classes)
	train_x = []
	train_y = []
	test_x = []
	test_y = []
	for i in range(num_classes):
		print("Start: " + classes[i])
		class_matrix = np.zeros(num_classes)
		class_matrix[i] = 1
		traindir = os.path.join(datapath, classes[i], "train")
		trainfiles = os.listdir(traindir)
		for f in trainfiles:
			train_x.append(pr.read_points(os.path.join(traindir, f), args.num_points_in))
			train_y.append(class_matrix)
		testdir = os.path.join(datapath, classes[i], "test")
		testfiles = os.listdir(testdir)
		for f in testfiles:
			test_x.append(pr.read_points(os.path.join(testdir, f), args.num_points_in))
			test_y.append(class_matrix)
		print("Done: " + classes[i])

	train_x = np.asarray(train_x)
	train_y = np.asarray(train_y)
	test_x = np.asarray(test_x)
	test_y = np.asarray(test_y)
	np.save(os.path.join(savepath, args.dataset + "_" + str(args.num_points_in) + "_train_x"), train_x)
	np.save(os.path.join(savepath, args.dataset + "_" + str(args.num_points_in) + "_train_y"), train_y)
	np.save(os.path.join(savepath, args.dataset + "_" + str(args.num_points_in) + "_test_x"), test_x)
	np.save(os.path.join(savepath, args.dataset + "_" + str(args.num_points_in) + "_test_y"), test_y)


if __name__ == "__main__":
	main()
