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
    anglemodulated bat algorithm (AMBA) was proposed for high-dimensional
    binary optimization problems with engineering application to antenna
    topology optimization. Note that ths problem has known exact solution -3103.

    The result in console should looks like this:

    protes > m 1.0e+02 | t 3.021e+00 | y -2.7560e+03
    protes > m 3.0e+02 | t 3.051e+00 | y -2.8150e+03
    protes > m 4.0e+02 | t 3.061e+00 | y -2.8350e+03
    protes > m 8.0e+02 | t 3.099e+00 | y -2.8700e+03
    protes > m 1.0e+03 | t 3.116e+00 | y -2.8850e+03
    protes > m 1.1e+03 | t 3.124e+00 | y -2.9070e+03
    protes > m 1.3e+03 | t 3.139e+00 | y -2.9350e+03
    protes > m 1.4e+03 | t 3.147e+00 | y -2.9690e+03
    protes > m 1.7e+03 | t 3.171e+00 | y -2.9990e+03
    protes > m 2.0e+03 | t 3.194e+00 | y -3.0030e+03
    protes > m 2.2e+03 | t 3.210e+00 | y -3.0700e+03
    protes > m 6.9e+03 | t 3.574e+00 | y -3.0720e+03
    protes > m 8.5e+03 | t 3.701e+00 | y -3.0750e+03
    protes > m 1.0e+04 | t 3.816e+00 | y -3.0750e+03 <<< DONE

    RESULT | y opt = -3.0750e+03 | time =     3.8277

    """
    d, n, f = func_build() # Target function, and array shape
    m = int(1.E+4)         # Number of requests to the objective function

    t = tpc()
    i_opt, y_opt = protes(f, d, n, m, log=True)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}')


if __name__ == '__main__':
    demo()
