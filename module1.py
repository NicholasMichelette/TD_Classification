import os
import sys
import numpy as np
import concurrent.futures
#import psutil
import time
#from pyntcloud import PyntCloud as pc
if __name__ == '__main__':
    import pyntcloud as pc
import pointreader as pr
import struct
import nltk
import matplotlib.pyplot as plt

#print(np.load('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/data_raw/bench_0001.npy')[12])

#def gcd(pair):
#    a, b = pair
#    low = min(a, b)
#    for i in range(low, 0, -1):
#        if a % i == 0 and b % i == 0:
#            return i

#numbers = [(1963309, 2265973), (2030677, 3814172), (1551645, 2229620), (2039045, 2020802)]
#def main():
#    for i in range(10):
#        start = time.time()
#        pool = concurrent.futures.ThreadPoolExecutor()
#        results = list(pool.map(gcd, numbers))
#        end = time.time()
#        print('Took %.3f seconds' % (end - start))

#def test(i):
#    import numpy as np
#    #import pyntcloud as pc
#    print(np.load('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/data_raw/bench_0001.npy')[12])


#def main():
#    pool = concurrent.futures.ProcessPoolExecutor()
#    for i in range(10):
#        #print("Queue: ", i)
#        pool.submit(test, i)
#    pool.shutdown()

#if __name__ == '__main__':
#    main()


#print(pr.read_points('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/data_raw/sydney/4wd/train/4wd.0.2299.csv', 1024).shape)

#t0 = time.time()
#file = open('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/data_raw/scanobjectnn/bed/train/scene0236_00_00001.bin', 'rb')
#contents = file.read()
#data = struct.unpack('f'*((len(contents))//4), contents)
#num_points = int(data[0])
#points = []
#for i in range(num_points):
#    points.append([data[i*11+1], data[i*11+2], data[i*11+3]])
#t1 = time.time()
#print(t1-t0)

#nltk.download('wordnet')
#sad = os.listdir(os.path.join(os.getcwd(), "data_raw", "shapenet"))
#for s in sad:
#    print(s, ": ", nltk.corpus.wordnet.synset_from_pos_and_offset('n', int(s)))

#ds = ["train_x", "train_y"]
#dets = ["ModelNet10_12_train_x", "ModelNet10_12_train_y"]
#dets2 = ["ModelNet10_12_test_", "ModelNet10_12_test_y"]
#for d in dets:
#    if (not ds[0] in d) and (not ds[1] in d):
#        print("NO")


#vox = np.load('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/Voxel/data/ModelNet10/toilet/train/toilet_0001___12.npy')
#for v in vox:
#    fig = plt.figure()
#    ax = plt.axes(projection = '3d')
#    ax.grid(True)
#    ti = time.time()
#    t = np.rot90(v, k=1, axes=(0, 1))
#    ax.voxels(t)
    
#    plt.show()


points = np.load('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/Point/data/ModelNet40/temp/chair_1024_test_x.npy')
ax = plt.axes(projection = '3d')
x = []
y = []
z = []
p = points[0]
for coord in p:
    x.append(coord[0])
    y.append(coord[1])
    z.append(coord[2])
ax.scatter3D(x, y, z)
plt.show()