import numpy as np
import os
from construct_TT import tens
import teneva


def gen_func_pair(num_ones=3):
    def f0(x):
        if x == 0 or x == num_ones:
            return 0

    def f1(x):
        return min(num_ones, x + 1)

    return [f0, f1]


def gen_func_pair_last(num_ones=3):
    def f0(x):
        if x == 0 or x == num_ones:
            return 1

    def f1(x):
        if x >= num_ones - 1:
            return 1

    return [f0, f1]



def ind_tens_max_ones(d, num_ones, r):
    funcs = [gen_func_pair(num_ones)]*(d-1) +  [gen_func_pair_last(num_ones)]
    cores = tens(funcs).cores
    update_to_rank_r(cores, r, noise=0, inplace=True)
    #cores = teneva.orthogonalize(cores, k=0)
    return cores


def update_to_rank_r(cores, r, noise=1e-3, inplace=False):
    d = len(cores)
    res = cores if inplace else [None]*d
    to_truncate = False
    for i, Y in enumerate(cores):
        r1, n, r2 = Y.shape
        nr1 = 1 if i==0     else r
        nr2 = 1 if i==(d-1) else r
        if nr1 < r1 or nr2 < r2:
            print("Initial: Order to reduce rank, so I'll truncate it. BAD")
            to_truncate = True

        if nr1 == r1 and nr2 == r2:
            res[i] = Y
            continue

        new_core = noise*np.random.random([max(nr1, r1), n, max(nr2, r2)])
        new_core[:r1, :, :r2] = Y

        res[i] = new_core

    if to_truncate:
        res = teneva.truncate(res, r=r)

    return res
