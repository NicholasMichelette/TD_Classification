import pathlib
import numpy as np
import random
import math
from pyntcloud import PyntCloud as pc


def read_points(filepath):
    test = pc.from_file(filepath)
    vert = test.get_sample(name='mesh_random', n=50000).to_numpy()
    return vert
    f = open(filepath, 'r')
    filetype = pathlib.Path(filepath).suffix
    verts = []
    max = [0, 0, 0]
    min = [0, 0, 0]
    if filetype == '.off':
        f.readline()
        firstline = f.readline().strip().split(' ')
        num_verts = int(firstline[0])
        num_faces = int(firstline[1])
        for i in range(num_verts):
            points_str = f.readline().strip().split(' ')
            point = []
            for coord in points_str:
                point.append(float(coord))
            if i == 0:
                max = point.copy()
                min = point.copy()
            else:
                for j in range(len(point)):
                    if point[j] > max[j]:
                        max[j] = point[j]
                    elif point[j] < min[j]:
                        min[j] = point[j]
            verts.append(point)
        print(len(verts))
        ran = np.subtract(np.asarray(max), np.asarray(min))
        mag = np.sqrt(ran.dot(ran))
        print('ran: ', ran)
        print('mag: ', mag)
        for i in range(num_faces):
            indices = []
            p = []
            line = f.readline().strip().split(' ')
            for l in range(1, len(line)):
                indices.append(int(line[l]))

            #max_sq_dist = 0
            #for j in range(len(indices) - 1):
            #    p1 = np.asarray(verts[indices[j]])
            #    for k in range(j + 1, len(indices)):
            #        p2 = np.asarray(verts[indices[k]])
            #        sq_dist = np.sum((p1-p2)**2, axis=0)
            #        if sq_dist > max_sq_dist:
            #            max_sq_dist = sq_dist
            #print(np.sqrt(max_sq_dist))
            for j in indices:
                p.append(verts[j])
            verts.extend(get_inbetween_points(p, mag))
    print(len(verts))
    return vert
    #return np.asarray(verts)


def get_inbetween_points(points, ran):
    ipoints = []
    
    if len(points) == 3:
        max_sq_dist = 0
        for j in range(len(points) - 1):
            p1 = np.asarray(points[j])
            for k in range(j + 1, len(points)):
                p2 = np.asarray(points[k])
                sq_dist = np.sum((p1-p2)**2, axis=0)
                if sq_dist > max_sq_dist:
                    max_sq_dist = sq_dist
        num_new_points = int((np.sqrt(max_sq_dist) / ran) * 100)
        for i in range(num_new_points):
            a = random.random()
            b = random.random()
            p1 = np.multiply(np.asarray(points[0]), 1 - math.sqrt(a))
            p2 = np.multiply(np.asarray(points[1]), math.sqrt(a) * (1 - b))
            p3 = np.multiply(np.asarray(points[2]), b * math.sqrt(a))
            ipoints.append(np.add(np.add(p1, p2), p3).tolist())
    elif len(points) == 4:
        p1 = []
        p2 = []
        p1.append(points[0])
        p1.append(points[1])
        p1.append(points[2])
        p2.append(points[0])
        p2.append(points[2])
        p2.append(points[3])
        ipoints = get_inbetween_points(p1).extend(get_inbetween_points(p2))
    return ipoints

