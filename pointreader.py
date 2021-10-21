import pathlib
import numpy as np
import random
import math
from pyntcloud import PyntCloud as pc


def read_points(filepath):
    test = pc.from_file(filepath)
    vert = test.get_sample(name='mesh_random', n=25000).to_numpy()
    return vert

