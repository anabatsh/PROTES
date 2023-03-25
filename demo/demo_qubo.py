import numpy as np
from time import perf_counter as tpc


from protes import protes


def func_build():
    """Binary knapsack problem."""

    d = 50
    n = 2

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
        """Target function: y=f(i); [d] -> float."""
        cost = np.dot(p, i)
        constr = np.dot(w, i)
        return 0 if constr > C else -cost

    return d, n, lambda I: np.array([func(i) for i in I])


def demo():
    """Demonstration for QUBO problem.

    We will solve the binary knapsack problem with fixed weights wi in [5, 20],
    profits pi in [50, 100] (i = 1, 2, . . . , d) and the maximum capacity
    C = 1000. It is from work (Dong et al., 2021) (problem k3; d = 50), where
    anglemodulated bat algorithm (AMBA) algorithm was proposed for
    high-dimensional binary optimization problems with engineering application
    to antenna topology optimization. Note that ths problem has known exact
    solution y_min = -3103.

    The result in console should looks like this:

    protes > m 5.0e+01 | t 3.371e+00 | y -2.5310e+03
    protes > m 1.0e+02 | t 3.495e+00 | y -2.7350e+03
    protes > m 2.5e+02 | t 3.827e+00 | y -2.8120e+03
    protes > m 5.0e+02 | t 4.367e+00 | y -2.8650e+03
    protes > m 2.0e+03 | t 7.821e+00 | y -2.9390e+03
    protes > m 3.2e+03 | t 1.044e+01 | y -2.9950e+03
    protes > m 3.7e+03 | t 1.152e+01 | y -3.0090e+03
    protes > m 4.4e+03 | t 1.302e+01 | y -3.0110e+03
    protes > m 5.0e+03 | t 1.422e+01 | y -3.0410e+03
    protes > m 6.2e+03 | t 1.699e+01 | y -3.0510e+03
    protes > m 6.6e+03 | t 1.765e+01 | y -3.0610e+03
    protes > m 7.4e+03 | t 1.966e+01 | y -3.0630e+03
    protes > m 8.0e+03 | t 2.088e+01 | y -3.0660e+03
    protes > m 8.2e+03 | t 2.120e+01 | y -3.0740e+03
    protes > m 8.2e+03 | t 2.144e+01 | y -3.0830e+03
    protes > m 8.5e+03 | t 2.201e+01 | y -3.0840e+03
    protes > m 8.6e+03 | t 2.212e+01 | y -3.0950e+03
    protes > m 1.1e+04 | t 2.643e+01 | y -3.0980e+03
    protes > m 2.0e+04 | t 4.734e+01 | y -3.0980e+03 <<< DONE

    RESULT | y opt = -3.09800e+03 | time = 47.350111849

    """
    d, n, f = func_build() # Target function, and array shape
    m = int(2.E+4)         # Number of requests to the objective function

    t = tpc()
    i_opt, y_opt = protes(f, d, n, m, log=True)
    print(f'\nRESULT | y opt = {y_opt:-11.5e} | time = {tpc()-t}')


if __name__ == '__main__':
    demo()
