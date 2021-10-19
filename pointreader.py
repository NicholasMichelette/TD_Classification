import pathlib
import numpy as np
import random
import math

def read_points(filepath):
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
            verts.append(point)


        for i in range(num_faces):
            indices = []
            p = []
            line = f.readline().strip().split(' ')
            for l in range(1, len(line)):
                indices.append(int(line[l]))
            for i in indices:
                p.append(verts[i])
            verts.extend(get_inbetween_points(p))

    return np.asarray(verts)


def get_inbetween_points(points):
    ipoints = []
    if len(points) == 3:
        for i in range(3):
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

