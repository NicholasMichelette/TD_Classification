import os


def main():
	datapath = os.path.join(os.getcwd(), "data_raw", "shapenet")
	classes = os.listdir(datapath)
	for c in classes:
		classpath = os.path.join(datapath, c)
		objects = os.listdir(classpath)
		for i in range(len(objects)):
			modelnum = "%04d" % i
			os.replace(os.path.join(classpath, objects[i], "models", "model_normalized.obj"), os.path.join(classpath, c + "_" + modelnum + ".obj"))
			os.rmdir(os.path.join(classpath, objects[i], "models"))
			os.rmdir(os.path.join(classpath, objects[i]))
		

if __name__ == "__main__":
	main()