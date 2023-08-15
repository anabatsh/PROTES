import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from time import perf_counter as tpc


from protes import protes


def func_build(d, n):
    """Ackley function. See https://www.sfu.ca/~ssurjano/ackley.html."""

    a = -32.768         # Grid lower bound
    b = +32.768         # Grid upper bound

    par_a = 20.         # Standard parameter values for Ackley function
    par_b = 0.2
    par_c = 2.*np.pi

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        X = I / (n - 1) * (b - a) + a

        y1 = np.sqrt(np.sum(X**2, axis=1) / d)
        y1 = - par_a * np.exp(-par_b * y1)

        y2 = np.sum(np.cos(par_c * X), axis=1)
        y2 = - np.exp(y2 / d)

        y3 = par_a + np.exp(1.)

        return y1 + y2 + y3

    return func


def demo():
    """A simple demonstration for discretized multivariate analytic function.

    We will find the minimum of an implicitly given "d"-dimensional array
    having "n" elements in each dimension. The array is obtained from the
    discretization of an analytic function.

    The result in console should looks like this (note that the exact minimum
    of this function is y = 0 and it is reached at the origin of coordinates):

    protes > m 1.0e+02 | t 3.092e+00 | y  2.0224e+01
    protes > m 2.0e+02 | t 3.104e+00 | y  1.9040e+01
    protes > m 3.0e+02 | t 3.108e+00 | y  1.8706e+01
    protes > m 5.0e+02 | t 3.116e+00 | y  1.7740e+01
    protes > m 6.0e+02 | t 3.121e+00 | y  1.6648e+01
    protes > m 1.0e+03 | t 3.135e+00 | y  1.5434e+01
    protes > m 1.3e+03 | t 3.146e+00 | y  1.4398e+01
    protes > m 1.5e+03 | t 3.152e+00 | y  1.4116e+01
    protes > m 2.0e+03 | t 3.168e+00 | y  1.2658e+01
    protes > m 2.5e+03 | t 3.188e+00 | y  8.4726e+00
    protes > m 2.9e+03 | t 3.203e+00 | y  0.0000e+00
    protes > m 1.0e+04 | t 3.440e+00 | y  0.0000e+00 <<< DONE

    RESULT | y opt =  0.0000e+00 | time =     3.4521

    """
    d = 7                # Dimension
    n = 11               # Mode size
    m = int(1.E+4)       # Number of requests to the objective function
    f = func_build(d, n) # Target function, which defines the array elements

    t = tpc()
    i_opt, y_opt = protes(f, d, n, m, log=True)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}\n\n')


if __name__ == '__main__':
    demo()
