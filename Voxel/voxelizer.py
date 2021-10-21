import numpy as np
import sys
import os
import time
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import pointreader as pr
import matplotlib.pyplot as plt


def voxelize(points, size): # points must be in the format of a numpy array of arrays, so [[x y z], ...]
    coords = find_coord_range(points)
    max_coord = coords[0]
    min_coord = coords[1]
    points_norm = np.copy(points)

    for p in range(len(points_norm)):
        for c in range(len(points_norm[p])):
            points_norm[p, c] = ((points_norm[p, c] - min_coord[c]) / (max_coord[c]- min_coord[c])) * size

    points_norm = points_norm.astype(int)
    voxel_grid_tf = np.zeros((size, size, size), dtype=bool)
    for p in points_norm:
        p0 = max(0, min(p[0], 31))
        p1 = max(0, min(p[1], 31))
        p2 = max(0, min(p[2], 31))
        voxel_grid_tf[p0, p1, p2] = True
    
    return voxel_grid_tf


def voxelize_hit(points, size): # points must be in the format of a numpy array of arrays, so [[x y z], ...]
    coords = find_coord_range(points)
    max_coord = coords[0]
    min_coord = coords[1]

    points_norm = np.copy(points)

    for p in range(len(points_norm)):
        for c in range(len(points_norm[p])):
            points_norm[p, c] = ((points_norm[p, c] - min_coord[c]) / (max_coord[c]- min_coord[c])) * size

    points_norm = points_norm.astype(int)
    voxel_grid = np.zeros((size, size, size), dtype=int)
    for p in points_norm:
        voxel_grid[p[0], p[1], p[2]] += 1


def find_coord_range(points):
    max = np.zeros(3)
    min = np.zeros(3)
    max[0] = min[0] = points[0][0]
    max[1] = min[1] = points[0][1]
    max[2] = min[2] = points[0][2]
    for p in points:
        for i in range(len(p)):
            if p[i] > max[i]:
                max[i] = p[i]
            elif p[i] < min[i]:
                min[i] = p[i]
    r = []
    r.append(np.add(max, 0.001))
    r.append(np.add(min, -0.001))

    return r


def swap(pointcloud):
    newcloud = []
    for i in pointcloud:
        newcloud.append([i[1], i[0], i[2]])
    newcloud = np.asarray(newcloud)


# For testing purposes
if __name__ == "__main__":
    ti = time.time()
    #voxel_grid_tf = voxelize(pr.read_points('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/data_raw/ModelNet40/bench/train/bench_0001.off'), 32)
    voxel_grid_tf = voxelize(pr.read_points('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/data_raw/ModelNet40/toilet/train/toilet_0322.off'), 32)
    #voxel_grid_tf = voxelize(pr.read_points('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/data_raw/ModelNet40/airplane/train/airplane_0001.off'), 32)
    print(time.time() - ti)
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.grid(True)
    ti = time.time()
    t = np.rot90(voxel_grid_tf, k=1, axes=(0, 1))
    print(time.time() - ti)
    ax.voxels(t)
    
    plt.show()
    