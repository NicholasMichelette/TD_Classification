import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import pointreader as pr
import argparse
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Process, Pool, Manager
from concurrent.futures import ProcessPoolExecutor
import psutil
import concurrent
from pyntcloud import PyntCloud as pc



parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='Dataset to preprocess.')
parser.add_argument('--num_points_in', default=1024, type=int, help='Number of points to sample from the data.')
parser.add_argument('--parallel', default=False, type=bool, help='Allows for multiprocessing.')
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



def get_points(files):
	filepath, training = files
	return pr.read_points(filepath, args.num_points_in)

def get_points2(L, filepath, num_p):
	L.append(pr.read_points(filepath, num_p))


def main_parallel():
	#mp.set_start_method('spawn')
	datapath = os.path.join(os.path.dirname(os.getcwd()), "data_raw", args.dataset)
	savepath = os.path.join(os.getcwd(), "data", args.dataset)
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	classes = os.listdir(datapath)
	num_classes = len(classes)
	train_classes = []
	train_x = []
	train_y = []
	test_classes = []
	test_x = []
	test_y = []
	for i in range(num_classes):
		with Manager() as manager:
			pool = Pool(maxtasksperchild=10)
			train_L = manager.list()
			test_L = manager.list()
			dir = os.path.join(datapath, classes[i], "train")
			files = os.listdir(dir)
			filespath = []
			processes = []
			for f in files:
				#p = Process(target=get_points2, args=(train_L, os.path.join(dir, f), args.num_points_in))
				#p.start()
				#processes.append(p)
				#filespath.append(os.path.join(dir, f))
				pool.apply_async(get_points2, args=(train_L, os.path.join(dir, f), args.num_points_in))
				#print(len(pool._cache))
			#train_L = pool.map_async(get_points, filespath)
			dir = os.path.join(datapath, classes[i], "test")
			files = os.listdir(dir)
			filespath = []
			for f in files:
				#p = Process(target=get_points2, args=(test_L, os.path.join(dir, f), args.num_points_in))
				#p.start()
				#processes.append(p)
				#filespath.append((os.path.join(dir, f), False))
				pool.apply_async(get_points2, args=(test_L, os.path.join(dir, f), args.num_points_in))
				#print(len(pool._cache))
			#test_L = pool.map_async(get_points, filespath)
			pool.close()
			pool.join()
			#for p in processes:
			#	p.join()
			train_classes.append(train_L)
			test_classes.append(test_L)
			print("Done: " + classes[i])

	for i in range(num_classes):
		class_matrix = np.zeros(num_classes)
		class_matrix[i] = 1
		for verts in train_classes[i]:
			train_x.append(verts)
			train_y.append(class_matrix)
		for verts in test_classes[i]:
			train_x.append(verts)
			train_y.append(class_matrix)

	train_x = np.asarray(train_x)
	train_y = np.asarray(train_y)
	test_x = np.asarray(test_x)
	test_y = np.asarray(test_y)
	np.save(os.path.join(savepath, args.dataset + "_" + str(args.num_points_in) + "_train_x"), train_x)
	np.save(os.path.join(savepath, args.dataset + "_" + str(args.num_points_in) + "_train_y"), train_y)
	np.save(os.path.join(savepath, args.dataset + "_" + str(args.num_points_in) + "_test_x"), test_x)
	np.save(os.path.join(savepath, args.dataset + "_" + str(args.num_points_in) + "_test_y"), test_y)
	

def get_points3(filepath):
	#test = pc.from_file(filepath)
	#vert = test.get_sample(name='mesh_random', n=args.num_points).to_numpy()
	#return vert
	print(os.path.basename(filepath))
	return pr.read_points(filepath, args.num_points_in)


def main_para():
	datapath = os.path.join(os.path.dirname(os.getcwd()), "data_raw", args.dataset)
	savepath = os.path.join(os.getcwd(), "data", args.dataset)
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	classes = os.listdir(datapath)
	num_classes = len(classes)
	train_classes = []
	train_x = []
	train_y = []
	test_classes = []
	test_x = []
	test_y = []
	with ProcessPoolExecutor() as exec:
		for i in range(num_classes):
			train_L = []
			test_L = []
			dir = os.path.join(datapath, classes[i], "train")
			files = os.listdir(dir)
			filespath = []
			for f in files:
				filespath.append(os.path.join(dir, f))
			train_L.append([r for r in exec.map(get_points3, filespath)])
			dir = os.path.join(datapath, classes[i], "test")
			files = os.listdir(dir)
			filespath = []
			for f in files:
				filespath.append(os.path.join(dir, f))
			test_L.append(r for r in exec.map(get_points3, filespath))
				
			train_classes.append(train_L)
			test_classes.append(test_L)
			print("Done: " + classes[i])
		print(exec)


	for i in range(num_classes):
		class_matrix = np.zeros(num_classes)
		class_matrix[i] = 1
		for verts in train_classes[i]:
			train_x.append(verts)
			train_y.append(class_matrix)
		for verts in test_classes[i]:
			train_x.append(verts)
			train_y.append(class_matrix)

	train_x = np.asarray(train_x)
	train_y = np.asarray(train_y)
	test_x = np.asarray(test_x)
	test_y = np.asarray(test_y)
	np.save(os.path.join(savepath, args.dataset + "_" + str(args.num_points_in) + "_train_x"), train_x)
	np.save(os.path.join(savepath, args.dataset + "_" + str(args.num_points_in) + "_train_y"), train_y)
	np.save(os.path.join(savepath, args.dataset + "_" + str(args.num_points_in) + "_test_x"), test_x)
	np.save(os.path.join(savepath, args.dataset + "_" + str(args.num_points_in) + "_test_y"), test_y)



if __name__ == "__main__":
	t0 = time.time()
	if args.parallel:
		main_para()
	else:
		main()
	t1 = time.time()
	print(t1-t0)
