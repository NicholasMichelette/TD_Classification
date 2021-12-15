import pathlib
import numpy as np
import random
import math
import time
import csv
import os
import random
import struct
from pyntcloud import PyntCloud as pc


PC_ACCEPTED = ['.npy', '.npz', '.las', '.obj', '.off', '.pcd', '.ply']

def read_points(filepath, num_points=25000):
    path, filetype = os.path.splitext(filepath)
    vert = []
    if filetype in PC_ACCEPTED:
        test = pc.from_file(filepath)
        vert = test.get_sample(name='mesh_random', n=num_points).to_numpy()
    elif "sydney" in path:
        file = open(filepath, 'r')
        csv_reader = csv.reader(file)
        temp = []
        for file in csv_reader:
            temp.append([float(file[3]), float(file[4]), float(file[5])])
        vert = vert + temp
        if num_points < 5000:
            while len(vert) < num_points:
                vert = vert + temp
        random.shuffle(vert)
        vert = vert[0:num_points]
        vert = np.asarray(vert)
        #file.close()
    elif "scanobjectnn" in path:
        file = open(filepath, 'rb')
        contents = file.read()
        data = struct.unpack('f'*((len(contents))//4), contents)
        num_p = int(data[0])
        for i in range(num_p):
            vert.append([data[i*11+1], data[i*11+2], data[i*11+3]])
        random.shuffle(vert)
        vert = vert[0:num_points]
        vert = np.asarray(vert)
        file.close()
    return vert

