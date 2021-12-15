import numpy as np
import os
import csv
import argparse
import random
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Name of dataset to use to test on. Must have already run preprocess.py(should have been run when training).")
parser.add_argument("--num_points_in", type=int, default=1024, help="Number of points that the dataset uses.")
parser.add_argument("--eval", type=bool, default=True, help="Evaluate all models and print them.")
parser.add_argument("--training", type=bool, default=True, help="Evaluate all models and print them.")
parser.add_argument("--validation", type=bool, default=True, help="Evaluate all models and print them.")
parser.add_argument("--top", type=int, default=0, help="Put only top n models in graph. ")
args = parser.parse_args()


def main():
	modelpath = os.path.join(os.getcwd(), "models")
	logpath = os.path.join(os.getcwd(), "logs")
	mn = os.listdir(modelpath)
	modelnames = []
	for n in mn:
		if args.dataset in n:
			modelnames.append(n)

	train_acc = []
	val_acc = []
	epoch = []

	for mname in modelnames:
		logname = mname + ".csv"
		val_logname = "val_" + logname
		path = os.path.join(logpath, logname)
		valpath = os.path.join(logpath, val_logname)
		if os.path.exists(path):
			f =  open(path, 'r')
			csv_reader = csv.reader(f)
			file_data = []
			for line in csv_reader:
				file_data.append(line[1])
			for i in range(1, len(file_data)):
				train_acc.append(float(file_data[i]))
				epoch.append(i)

			f =  open(valpath, 'r')
			csv_reader = csv.reader(f)
			file_data = []
			for line in csv_reader:
				file_data.append(line[1])
			for i in range(1, len(file_data)):
				val_acc.append(float(file_data[i]))

	plt.title(args.dataset)
	plt.plot(epoch, val_acc, 'b', label="val")
	if args.training:
		plt.plot(epoch, train_acc, 'r', label="training")
	plt.show()
		

if __name__ == "__main__":
	main()