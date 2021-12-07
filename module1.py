import numpy as np
import concurrent.futures
import psutil
import os
import time

#print(np.load('J:/Documents/School/2021 fall/ML2/Project/TD_Classification/data_raw/bench_0001.npy')[12])

def gcd(pair):
    a, b = pair
    low = min(a, b)
    for i in range(low, 0, -1):
        if a % i == 0 and b % i == 0:
            return i

numbers = [(1963309, 2265973), (2030677, 3814172),
           (1551645, 2229620), (2039045, 2020802)]

def main():
    for i in range(1000000):


    for i in range(10):
        start = time.time()
        pool = concurrent.futures.ProcessPoolExecutor()
        results = list(pool.map(gcd, numbers))
        end = time.time()
        print('Took %.3f seconds' % (end - start))


if __name__ == '__main__':
    main()
