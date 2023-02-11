import numpy as np
import sys
from time import perf_counter as tpc


from protes import protes


def func_build(d, n, mod='jax'):
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
        """Target function: y=f(i); [d] -> float."""
        if mod == 'tor':
            i = i.numpy()

        cost = np.dot(p, i)
        constr = np.dot(w, i)
        return 0 if constr > C else -cost

    return lambda I: np.array([func(i) for i in I])


def demo(mod):
    """Demonstration for QUBO problem.

    We will solve the binary knapsack problem with fixed weights wi in [5, 20],
    profits pi in [50, 100] (i = 1, 2, . . . , d) and the maximum capacity
    C = 1000. It is from work (Dong et al., 2021) (problem k3; d = 50), where
    anglemodulated bat algorithm (AMBA) algorithm was proposed for
    high-dimensional binary optimization problems with engineering application
    to antenna topology optimization. Note that ths problem has known exact
    solution y_min = -3103.

    The result in console then "mod" is "jax" should looks like this:

    protes-jax > m 5.0e+01 | t 3.109e+00 | y -2.5310e+03
    protes-jax > m 1.0e+02 | t 3.231e+00 | y -2.7350e+03
    protes-jax > m 2.5e+02 | t 3.586e+00 | y -2.8120e+03
    protes-jax > m 5.0e+02 | t 4.134e+00 | y -2.8650e+03
    protes-jax > m 2.0e+03 | t 7.526e+00 | y -2.9390e+03
    protes-jax > m 3.2e+03 | t 1.018e+01 | y -2.9950e+03
    protes-jax > m 3.7e+03 | t 1.133e+01 | y -3.0090e+03
    protes-jax > m 4.4e+03 | t 1.288e+01 | y -3.0110e+03
    protes-jax > m 5.0e+03 | t 1.409e+01 | y -3.0410e+03
    protes-jax > m 6.2e+03 | t 1.692e+01 | y -3.0510e+03
    protes-jax > m 6.6e+03 | t 1.756e+01 | y -3.0610e+03
    protes-jax > m 7.4e+03 | t 1.948e+01 | y -3.0630e+03
    protes-jax > m 8.0e+03 | t 2.065e+01 | y -3.0660e+03
    protes-jax > m 8.2e+03 | t 2.097e+01 | y -3.0740e+03
    protes-jax > m 8.2e+03 | t 2.119e+01 | y -3.0830e+03
    protes-jax > m 8.5e+03 | t 2.175e+01 | y -3.0840e+03
    protes-jax > m 8.6e+03 | t 2.186e+01 | y -3.0950e+03
    protes-jax > m 1.1e+04 | t 2.627e+01 | y -3.0980e+03
    protes-jax > m 2.0e+04 | t 4.660e+01 | y -3.0980e+03 <<< DONE

    The result in console then "mod" is "tor" should looks like this:

    protes-tor > m 5.0e+01 | t 4.503e+00 | y -2.7220e+03
    protes-tor > m 1.0e+02 | t 8.885e+00 | y -2.7640e+03
    protes-tor > m 5.0e+02 | t 4.263e+01 | y -2.7790e+03
    protes-tor > m 1.0e+03 | t 8.470e+01 | y -2.8120e+03
    protes-tor > m 1.5e+03 | t 1.272e+02 | y -2.8710e+03
    protes-tor > m 2.2e+03 | t 1.831e+02 | y -2.8960e+03
    protes-tor > m 2.9e+03 | t 2.475e+02 | y -2.9060e+03
    protes-tor > m 3.0e+03 | t 2.562e+02 | y -2.9680e+03
    protes-tor > m 5.0e+03 | t 4.226e+02 | y -2.9760e+03
    protes-tor > m 5.2e+03 | t 4.486e+02 | y -2.9860e+03
    protes-tor > m 5.3e+03 | t 4.529e+02 | y -3.0100e+03
    protes-tor > m 5.4e+03 | t 4.617e+02 | y -3.0240e+03
    protes-tor > m 5.8e+03 | t 4.960e+02 | y -3.0430e+03
    protes-tor > m 6.8e+03 | t 5.829e+02 | y -3.0490e+03
    protes-tor > m 7.1e+03 | t 6.080e+02 | y -3.0550e+03
    protes-tor > m 7.4e+03 | t 6.289e+02 | y -3.0690e+03
    protes-tor > m 8.8e+03 | t 7.575e+02 | y -3.0880e+03
    protes-tor > m 1.0e+04 | t 8.696e+02 | y -3.0900e+03
    protes-tor > m 1.0e+04 | t 8.739e+02 | y -3.0930e+03
    protes-tor > m 1.2e+04 | t 9.982e+02 | y -3.0940e+03
    protes-tor > m 1.2e+04 | t 1.032e+03 | y -3.0960e+03
    protes-tor > m 1.3e+04 | t 1.096e+03 | y -3.1030e+03
    protes-tor > m 2.0e+04 | t 1.697e+03 | y -3.1030e+03 <<< DONE

    """
    d = 50               # Dimension
    n = 2                # Mode size
    m = int(2.E+4)       # Number of requests to the objective function
    f = func_build(d, n, mod)

    t = tpc()
    i_opt, y_opt = protes(f, [n]*d, m, mod=mod, log=True)
    print(f'\nRESULT | y opt = {y_opt:-11.5e} | time = {tpc()-t}')


if __name__ == '__main__':
    np.random.seed(42)

    mod = sys.argv[1] if len(sys.argv) > 1 else 'jax'
    demo(mod)
