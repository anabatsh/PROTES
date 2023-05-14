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

    protes > m 5.0e+01 | t 3.038e+00 | y -2.5310e+03
    protes > m 1.0e+02 | t 3.169e+00 | y -2.7350e+03
    protes > m 2.5e+02 | t 3.601e+00 | y -2.8120e+03
    protes > m 5.0e+02 | t 4.231e+00 | y -2.8650e+03
    protes > m 2.0e+03 | t 8.118e+00 | y -2.9390e+03
    protes > m 3.2e+03 | t 1.120e+01 | y -2.9950e+03
    protes > m 3.7e+03 | t 1.249e+01 | y -3.0090e+03
    protes > m 4.6e+03 | t 1.474e+01 | y -3.0190e+03
    protes > m 4.7e+03 | t 1.498e+01 | y -3.0310e+03
    protes > m 5.0e+03 | t 1.559e+01 | y -3.0410e+03
    protes > m 6.0e+03 | t 1.823e+01 | y -3.0730e+03
    protes > m 7.1e+03 | t 2.097e+01 | y -3.0860e+03
    protes > m 9.2e+03 | t 2.618e+01 | y -3.0920e+03
    protes > m 1.2e+04 | t 3.336e+01 | y -3.0950e+03
    protes > m 2.0e+04 | t 5.285e+01 | y -3.0950e+03 <<< DONE

    RESULT | y opt = -3.09500e+03 | time =    52.8616

    """
    d, n, f = func_build() # Target function, and array shape
    m = int(2.E+4)         # Number of requests to the objective function

    t = tpc()
    i_opt, y_opt = protes(f, d, n, m, log=True)
    print(f'\nRESULT | y opt = {y_opt:-11.5e} | time = {tpc()-t:-10.4f}')


if __name__ == '__main__':
    demo()
