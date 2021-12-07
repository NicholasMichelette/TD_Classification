import pathlib
import numpy as np
import random
import math
import time
from pyntcloud import PyntCloud as pc


def read_points(filepath, num_points=25000):
    test = pc.from_file(filepath)
    vert = test.get_sample(name='mesh_random', n=num_points).to_numpy()
    return vert

