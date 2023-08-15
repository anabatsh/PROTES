import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from time import perf_counter as tpc


from protes import protes_general


def func_build(d, n):
    """Ackley function. See https://www.sfu.ca/~ssurjano/ackley.html."""

    a = -32.768         # Grid lower bound
    b = +32.768         # Grid upper bound

    par_a = 20.         # Standard parameter values for Ackley function
    par_b = 0.2
    par_c = 2.*np.pi

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        X = I / (np.array(n) - 1) * (b - a) + a

        y1 = np.sqrt(np.sum(X**2, axis=1) / d)
        y1 = - par_a * np.exp(-par_b * y1)

        y2 = np.sum(np.cos(par_c * X), axis=1)
        y2 = - np.exp(y2 / d)

        y3 = par_a + np.exp(1.)

        return y1 + y2 + y3

    return func


def demo():
    """A simple demonstration for "ptotes_general" method.

    We will find the minimum of an implicitly given "d"-dimensional array
    having various number of elements in each dimension with "ptotes_general".
    We consider the same function as in the "demo_func.py" example.

    The result in console should looks like this (note that the exact minimum
    of this function is y = 0 and it is reached at the origin of coordinates):

    protes > m 1.0e+02 | t 4.034e+00 | y  1.9114e+01
    protes > m 2.0e+02 | t 4.047e+00 | y  1.5828e+01
    protes > m 4.0e+02 | t 4.056e+00 | y  1.2533e+01
    protes > m 5.0e+02 | t 4.060e+00 | y  8.6142e+00
    protes > m 1.1e+03 | t 4.081e+00 | y  0.0000e+00
    protes > m 2.0e+03 | t 4.109e+00 | y  0.0000e+00 <<< DONE

    RESULT | y opt =  0.0000e+00 | time =     4.1281

    """
    d = 5                  # Dimension
    n = [5, 7, 9, 11, 13]  # Mode sizes
    m = int(2.E+3)         # Number of requests to the objective function
    f = func_build(d, n)   # Target function, which defines the array elements

    t = tpc()
    i_opt, y_opt = protes_general(f, n, m, log=True)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}\n\n')


if __name__ == '__main__':
    demo()
