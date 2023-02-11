import numpy as np
import sys
from time import perf_counter as tpc


from protes import protes


def func_build(d, n, mod='jax'):
    """Ackley function. See https://www.sfu.ca/~ssurjano/ackley.html."""

    a = -32.768         # Grid lower bound
    b = +32.768         # Grid upper bound

    par_a = 20.         # Standard parameter values for Ackley function
    par_b = 0.2
    par_c = 2.*np.pi

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        if mod == 'tor':
            I = I.numpy()

        X = I / (n - 1) * (b - a) + a

        y1 = np.sqrt(np.sum(X**2, axis=1) / d)
        y1 = - par_a * np.exp(-par_b * y1)

        y2 = np.sum(np.cos(par_c * X), axis=1)
        y2 = - np.exp(y2 / d)

        y3 = par_a + np.exp(1.)

        return y1 + y2 + y3

    return func


def demo(mod):
    """A simple demonstration for discretized multivariate analytic function.

    We will find the minimum of an implicitly given "d"-dimensional tensor
    having "n" elements in each dimension. The tensor is obtained from the
    discretization of an analytic function.

    The result in console then "mod" is "jax" should looks like this:

    protes-jax > m 5.0e+01 | t 3.163e+00 | y  1.5171e+01
    protes-jax > m 2.6e+03 | t 7.712e+00 | y  1.2532e+01
    protes-jax > m 4.9e+03 | t 1.217e+01 | y  1.2532e+01
    protes-jax > m 1.0e+04 | t 2.139e+01 | y  1.2532e+01 <<< DONE

    The result in console then "mod" is "tor" should looks like this:

    protes-tor > m 5.0e+01 | t 6.110e-01 | y  1.9747e+01
    protes-tor > m 1.0e+02 | t 1.225e+00 | y  1.9664e+01
    protes-tor > m 2.0e+02 | t 2.324e+00 | y  1.9542e+01
    protes-tor > m 2.5e+02 | t 2.878e+00 | y  1.9542e+01
    protes-tor > m 5.5e+02 | t 6.290e+00 | y  1.8754e+01
    protes-tor > m 7.0e+02 | t 7.959e+00 | y  1.8118e+01
    protes-tor > m 9.0e+02 | t 1.024e+01 | y  1.7381e+01
    protes-tor > m 1.1e+03 | t 1.247e+01 | y  1.7381e+01
    protes-tor > m 1.3e+03 | t 1.466e+01 | y  1.5171e+01
    protes-tor > m 2.8e+03 | t 3.202e+01 | y  1.5171e+01
    protes-tor > m 3.2e+03 | t 3.657e+01 | y  1.2532e+01
    protes-tor > m 3.3e+03 | t 3.723e+01 | y  1.2532e+01
    protes-tor > m 4.9e+03 | t 5.516e+01 | y  1.2532e+01
    protes-tor > m 1.0e+04 | t 1.105e+02 | y  1.2532e+01 <<< DONE

    """
    d = 7                # Dimension
    n = 10               # Mode size
    m = int(1.E+4)       # Number of requests to the objective function
    f = func_build(d, n, mod)

    t = tpc()
    i_opt, y_opt = protes(f, [n]*d, m, mod=mod, log=True)
    print(f'\nRESULT | y opt = {y_opt:-11.5e} | time = {tpc()-t}')


if __name__ == '__main__':
    np.random.seed(42)

    mod = sys.argv[1] if len(sys.argv) > 1 else 'jax'
    demo(mod)
