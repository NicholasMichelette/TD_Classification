import os
import sys
import argparse
import numpy as np
import math
import time
from pathlib import Path
import voxelizer
import matplotlib.pyplot as plt
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import pointreader as pr


parser = argparse.ArgumentParser()
parser.add_argument('Dataset', type=str, help='Dataset to voxelize within the data_raw directory.')
parser.add_argument('num_points', default=25000, type=int, help='Number of points to sample from the data.')


def voxelize_data(dataset, num_points=25000, num_rotations=12, num_voxels=32):
    datapath = os.path.join(sys.path[0], "data")
    Path(datapath).mkdir(parents=True, exist_ok=True)
    datapath = os.path.join(datapath, dataset)
    Path(datapath).mkdir(parents=True, exist_ok=True)
    raw_datapath = os.path.join(sys.path[0].strip('/\Voxel'), 'data_raw', dataset)
    object_types = os.listdir(raw_datapath)
    for ot in object_types:
        rdp = os.path.join(raw_datapath, ot)
        dp = os.path.join(datapath, ot)
        Path(dp).mkdir(parents=True, exist_ok=True)
        test_train = os.listdir(rdp)
        for t in test_train:
            trdp = os.path.join(rdp, t)
            tdp = os.path.join(dp, t) 
            Path(tdp).mkdir(parents=True, exist_ok=True)
            data = os.listdir(trdp)
            completed = os.listdir(tdp)
            for d in data:
                rdat = os.path.join(trdp, d)
                dat = os.path.join(tdp, d)
                dat = dat.strip('.off') + '___' + str(num_rotations)
                da = d.strip('.off') + '___' + str(num_rotations) + '.npy'
                if da not in completed:
                    cloud = pr.read_points(rdat, num_points)
                    np.save(dat, get_rotations(cloud, num_rotations))
        print("Done: ", ot)


def get_rotations(pointcloud, num_rots, num_voxels=32):
    x_sum = 0
    y_sum = 0
    for p in pointcloud:
        x_sum += p[0]
        y_sum += p[1]
    center = np.asarray([x_sum/len(pointcloud), y_sum/len(pointcloud)])

    rotations = []
    rotations.append(pointcloud)
    for n in range(1, int(num_rots/4)):
        rot =[]
        ang = n * (2 * math.pi / num_rots)
        for i in pointcloud:
            x, y = rotate(i, center, math.cos(ang), math.sin(ang))
            rot.append([x, y, i[2]])
        rotations.append(np.asarray(rot))

    voxelizations = []
    for r in rotations:
        v = voxelizer.voxelize(r, num_voxels)
        voxelizations.append(v)
        voxelizations.append(np.flip(v, axis=1))
        n = np.rot90(v, k=1, axes=(0, 1))
        voxelizations.append(n)
        voxelizations.append(np.flip(n, axis=1))

    return voxelizations

    
def rotate(point, origin, cos, sin):
    x = point[0]
    y = point[1]
    ox = origin[0]
    oy = origin[1]
    ax = x - ox
    ay = y - oy
    qx = ox + cos * ax + sin * ay
    qy = oy - sin * ax + cos * ay
    return qx, qy


if __name__ == "__main__":
    args = parser.parse_args()
    voxelize_data(args.Dataset, num_points=args.num_points)
    
    #temp = pr.read_points('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/data_raw/ModelNet40/bench/train/bench_0001.off')
    #t = time.time()
    #v = get_rotations(temp, 12)
    #print(time.time() - t)
    #print(v)
    #np.save('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/data_raw/bench_0001.off'.strip('.off'), v)
    #fig = plt.figure()
    #ax = plt.axes(projection = '3d')
    #ax.voxels(v[8])  
    #plt.show()