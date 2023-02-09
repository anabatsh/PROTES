import numpy as np
from time import perf_counter as tpc


from protes import protes


def func_build(d, n):
    """Binary knapsack problem."""

    assert d == 50
    assert n == 2

    w = [
        80, 82, 85, 70, 72, 70, 66, 50, 55, 25, 50, 55, 40, 48, 59, 32, 22,
        60, 30, 32, 40, 38, 35, 32, 25, 28, 30, 22, 50, 30, 45, 30, 60, 50,
        20, 65, 20, 25, 30, 10, 20, 25, 15, 10, 10, 10, 4, 4, 2, 1]

    p = [
        220, 208, 198, 192, 180, 180, 165, 162, 160, 158, 155, 130, 125,
        122, 120, 118, 115, 110, 105, 101, 100, 100, 98, 96, 95, 90, 88, 82,
        80, 77, 75, 73, 72, 70, 69, 66, 65, 63, 60, 58, 56, 50, 30, 20, 15,
        10, 8, 5, 3, 1]

    C = 1000

    def func(i):
        cost = np.dot(p, i)
        constr = np.dot(w, i)
        return 0 if constr > C else -cost

    return lambda I: np.array([func(i) for i in I])


def demo():
    """Demonstration for QUBO problem.

    We will solve the binary knapsack problem with fixed weights wi in [5, 20],
    profits pi in [50, 100] (i = 1, 2, . . . , d) and the maximum capacity
    C = 1000. It is from work (Dong et al., 2021) (problem k3; d = 50), where
    anglemodulated bat algorithm (AMBA) algorithm was proposed for
    high-dimensional binary optimization problems with engineering application
    to antenna topology optimization. Note that ths problem has known exact
    solution y_min = -3103.

    """
    d = 50               # Dimension
    n = 2                # Mode size
    m = int(2.E+4)       # Number of requests to the objective function
    f = func_build(d, n) # Target function (y=f(I); [samples,d] -> [samples])

    t = tpc()
    i_opt, y_opt = protes(f, [n]*d, m, log=True)
    print(f'\nRESULT | y opt = {y_opt:-11.5e} | time = {tpc()-t}')


if __name__ == '__main__':
    np.random.seed(42)
    demo()
