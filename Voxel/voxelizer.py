import numpy as np
import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import pointreader as pr
import matplotlib.pyplot as plt


def voxelize(points, size): # points must be in the format of a numpy array of arrays, so [[x y z], ...]
    #max_coord = find_max_coord(points)
    coords = find_coord_range(points)
    max_coord = coords[0]
    min_coord = coords[1]

    points_norm = np.copy(points)

    for p in range(len(points_norm)):
        for c in range(len(points_norm[p])):
            points_norm[p, c] = ((points_norm[p, c] - min_coord[c]) / (max_coord[c]- min_coord[c])) * size


    #for p in range(len(points_norm)):
    #    for c in range(len(points_norm[p])):
    #        if p < 100:
    #            print(points_norm[p, c], ' ', points_norm[p], ' ', max_coord[c], ' ', c)
    #        points_norm[p, c] = ((points_norm[p, c] / max_coord[c] + 1) / 2 ) * size
            
    #points_norm = np.true_divide(np.add(np.true_divide(points, max_coord), 1), 2)
    #points_norm = np.multiply(points_norm, size)
    points_norm = points_norm.astype(int)
    voxel_grid = np.zeros((size, size, size), dtype=int)
    voxel_grid_tf = np.zeros((size, size, size), dtype=bool)
    for p in points_norm:
        voxel_grid[p[0], p[1], p[2]] += 1
        voxel_grid_tf[p[0], p[1], p[2]] = True
    



    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    x = []
    y = []
    z = []
    for i in range(len(voxel_grid)):
        for j in range(len(voxel_grid[i])):
            for k in range(len(voxel_grid[i, j])):
                if voxel_grid[i, j, k] != 0:
                    x.append(i)
                    y.append(j)
                    z.append(k)

    x1 = []
    y1 = [] 
    z1 = []
    #print(len(points))
    for i in range(len(points)):
        x1.append(points_norm[i, 0])
        y1.append(points_norm[i, 1])
        z1.append(points_norm[i, 2])

    #ax.scatter3D(x, y, z, c=z)
    #ax.scatter3D(x1, y1, z1, c=z1)
    ax.grid(True)
    ax.voxels(voxel_grid_tf)
    plt.show()


def find_max_coord(points):
    max = np.zeros(3)
    for p in points:
        for i in range(len(p)):
            if abs(p[i]) > max[i]:
                max[i] = abs(p[i])
    return np.add(max, 0.000001)

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

#voxelize(pr.read_points('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/data_raw/ModelNet40/bench/train/bench_0001.off'), 32)
#voxelize(pr.read_points('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/data_raw/ModelNet40/toilet/train/toilet_0322.off'), 32)
voxelize(pr.read_points('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/data_raw/ModelNet40/airplane/train/airplane_0001.off'), 32)