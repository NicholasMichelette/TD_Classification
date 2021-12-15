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
parser.add_argument("--training", type=bool, default=False, help="Evaluate all models and print them.")
parser.add_argument("--validation", type=bool, default=True, help="Evaluate all models and print them.")
parser.add_argument("--top", type=int, default=0, help="Put only top n models in graph. ")
args = parser.parse_args()


def get_accuracy(elem):
	# elem is formatted like [model_name, [loss, accuracy]]
	return elem[1][1]


def main():
	datapath = os.path.join(os.getcwd(), "data", args.dataset)
	logpath = os.path.join(os.getcwd(), "logs")
	modelpath = os.path.join(os.getcwd(), "models")
	log_evalpath = os.path.join(os.getcwd(), "log_eval")
	if not os.path.exists(log_evalpath):
		os.mkdir(log_evalpath)
	test_x_filename = os.path.join(datapath, args.dataset + "_" + str(args.num_points_in) + "_" + "test_x.npy")
	test_y_filename = os.path.join(datapath, args.dataset + "_" + str(args.num_points_in) + "_" + "test_y.npy")
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
	eval = []
	mn = os.listdir(modelpath)
	modelnames = []
	topn = []
	topn_names = []
	for n in mn:
		if args.dataset in n:
			modelnames.append(n)
	if args.eval:
		with tf.device('/GPU:0'):
			for mname in modelnames:
				eval_data = [mname]
				if os.path.exists(os.path.join(log_evalpath, mname + ".npy")):
					eval_data.append(np.load(os.path.join(log_evalpath, mname + ".npy"), allow_pickle=True)[1])
				else:
					model = load_model(os.path.join(modelpath, mname))
					eval_data.append(model.evaluate(test_x, test_y, batch_size=32, verbose=0))
					np.save(os.path.join(log_evalpath, mname), eval_data)
				eval.append(eval_data)
			eval.sort(reverse=True, key=get_accuracy)
		for e in eval:
			print(e[0], "	", e[1][1])
		if args.top != 0:
			topn = eval[:args.top]
			for t in topn:
				topn_names.append(t[0])

	logdata = []
	for mname in modelnames:
		logname = mname + ".csv"
		path = os.path.join(logpath, logname)
		if os.path.exists(path):
			fulllog = [mname]
			log = []
			f =  open(path, 'r')
			csv_reader = csv.reader(f)
			file_data = []
			for line in csv_reader:
				file_data.append([line[1], line[3]])
			for i in range(1, len(file_data)):
				log.append([i-1, file_data[i][0], file_data[i][1]])
			fulllog.append(log)
			logdata.append(fulllog)
	
	if len(topn) == 0:
		for log in logdata:
			x_axis_epoch = []
			y_axis_val_acc = []
			y_axis_train_acc = []
			for point in log[1]:
				x_axis_epoch.append(int(point[0]))
				y_axis_val_acc.append(float(point[2]))
				y_axis_train_acc.append(float(point[1]))
			r = random.random()
			g = random.random()
			b = random.random()
			color = (r, g, b)
			lab = log[0].split("1024")[-1][1:]
			if args.validation:
				plt.plot(x_axis_epoch, y_axis_val_acc, c=color, label=lab)
			if args.training:
				plt.plot(x_axis_epoch, y_axis_train_acc, c=color, label=lab+"_training")
	else:
		for log in logdata:
			if log[0] in topn_names:
				x_axis_epoch = []
				y_axis_val_acc = []
				y_axis_train_acc = []
				for point in log[1]:
					x_axis_epoch.append(int(point[0]))
					y_axis_val_acc.append(float(point[2]))
					y_axis_train_acc.append(float(point[1]))
				r = random.random()
				g = random.random()
				b = random.random()
				color = (r, g, b)
				lab = log[0].split("1024")[-1][1:]
				if args.validation:
					plt.plot(x_axis_epoch, y_axis_val_acc, c=color, label=lab)
				if args.training:
					if args.top == 1:
						color = 'r'
					plt.plot(x_axis_epoch, y_axis_train_acc, c=color, label=lab+"_training")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')
	plt.title(args.dataset)
	plt.show()

if __name__ == "__main__":
	main()
