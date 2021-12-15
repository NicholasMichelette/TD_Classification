import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Name of dataset to use to split. Must already be in categories, but not split into test and train sets.")
parser.add_argument("--split", type=float, default=0.1, help="Split of data that will be test data.")
args = parser.parse_args()


def main():
	datapath = os.path.join(os.getcwd(), "data_raw", args.dataset)
	classes = os.listdir(datapath)
	for c in classes:
		classpath = os.path.join(datapath, c)
		testpath = os.path.join(classpath, "test")
		trainpath = os.path.join(classpath, "train")
		files = os.listdir(testpath)
		for i in range(len(files)):
			os.replace(os.path.join(testpath, files[i]), os.path.join(classpath, files[i]))
		files = os.listdir(trainpath)
		for i in range(len(files)):
			os.replace(os.path.join(trainpath, files[i]), os.path.join(classpath, files[i]))
		os.removedirs(testpath)
		os.removedirs(trainpath)

if __name__ == "__main__":
	main()
