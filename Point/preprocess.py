import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import pointreader as pr
import pyntcloud as pc
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='Dataset to preprocess.')
parser.add_argument('--num_points_in', default=1024, type=int, help='Number of points to sample from the data.')
parser.add_argument('--debug', default=False, type=bool, help='Print files as they are processed.')
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
	savetemp = os.path.join(savepath, "temp")
	if not os.path.exists(savetemp):
		os.mkdir(savetemp)
	completed = os.listdir(savetemp)
	for i in range(len(completed)):
		completed[i] = completed[i].split(str(args.num_points_in))[0] + str(args.num_points_in)
	for i in range(num_classes):
		if classes[i] + "_" + str(args.num_points_in) in completed:
			print("Already Processed: " + classes[i])
		else:
			print("Start: " + classes[i])
			train_x_class = []
			train_y_class = []
			test_x_class = []
			test_y_class = []
			class_matrix = np.zeros(num_classes)
			class_matrix[i] = 1
			traindir = os.path.join(datapath, classes[i], "train")
			trainfiles = os.listdir(traindir)
			for f in trainfiles:
				if args.debug:
					print(f)
				train_x_class.append(pr.read_points(os.path.join(traindir, f), args.num_points_in))
				train_y_class.append(class_matrix)
			testdir = os.path.join(datapath, classes[i], "test")
			testfiles = os.listdir(testdir)
			for f in testfiles:
				if args.debug:
					print(f)
				test_x_class.append(pr.read_points(os.path.join(testdir, f), args.num_points_in))
				test_y_class.append(class_matrix)
			train_x_class = np.asarray(train_x_class)
			train_y_class = np.asarray(train_y_class)
			test_x_class = np.asarray(test_x_class)
			test_y_class = np.asarray(test_y_class)
			np.save(os.path.join(savetemp, classes[i] + "_" + str(args.num_points_in) + "_train_x"), train_x_class)
			np.save(os.path.join(savetemp, classes[i] + "_" + str(args.num_points_in) + "_train_y"), train_y_class)
			np.save(os.path.join(savetemp, classes[i] + "_" + str(args.num_points_in) + "_test_x"), test_x_class)
			np.save(os.path.join(savetemp, classes[i] + "_" + str(args.num_points_in) + "_test_y"), test_y_class)
			print("Done: " + classes[i])

	completed = os.listdir(savetemp)
	for file in completed:
		cd = np.load(os.path.join(savetemp, file))
		if str(args.num_points_in) + "_train_x" in file:
			for points in cd:
				train_x.append(points)
		if str(args.num_points_in) + "_train_y" in file:
			for point_class in cd:
				train_y.append(point_class)
		if str(args.num_points_in) + "_test_x" in file:
			for points in cd:
				test_x.append(points)
		if str(args.num_points_in) + "_test_y" in file:
			for point_class in cd:
				test_y.append(point_class)
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
