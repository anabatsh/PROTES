import jax.numpy as jnp
from time import perf_counter as tpc


from protes import protes


def func_build(d, n):
    """Ackley function. See https://www.sfu.ca/~ssurjano/ackley.html."""

    a = -32.768         # Grid lower bound
    b = +32.768         # Grid upper bound

    par_a = 20.         # Standard parameter values for Ackley function
    par_b = 0.2
    par_c = 2.*jnp.pi

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        X = I / (n - 1) * (b - a) + a

        y1 = jnp.sqrt(jnp.sum(X**2, axis=1) / d)
        y1 = - par_a * jnp.exp(-par_b * y1)

        y2 = jnp.sum(jnp.cos(par_c * X), axis=1)
        y2 = - jnp.exp(y2 / d)

        y3 = par_a + jnp.exp(1.)

        return y1 + y2 + y3

    return func


def demo():
    """A simple demonstration for discretized multivariate analytic function.

    We will find the minimum of an implicitly given "d"-dimensional array
    having "n" elements in each dimension. The array is obtained from the
    discretization of an analytic function.

    The result in console should looks like this (note that the exact minimum
    of this function is y = 0 and it is reached at the origin of coordinates):

    protes > m 1.0e+02 | t 3.190e+00 | y  2.0214e+01
    protes > m 2.0e+02 | t 3.203e+00 | y  1.8211e+01
    protes > m 5.0e+02 | t 3.216e+00 | y  1.8174e+01
    protes > m 6.0e+02 | t 3.220e+00 | y  1.7491e+01
    protes > m 7.0e+02 | t 3.224e+00 | y  1.7078e+01
    protes > m 8.0e+02 | t 3.228e+00 | y  1.6180e+01
    protes > m 1.1e+03 | t 3.238e+00 | y  1.4116e+01
    protes > m 1.4e+03 | t 3.250e+00 | y  8.4726e+00
    protes > m 2.7e+03 | t 3.293e+00 | y  0.0000e+00
    protes > m 1.0e+04 | t 3.534e+00 | y  0.0000e+00 <<< DONE

    RESULT | y opt =  0.0000e+00 | time =     3.5459

    """
    d = 7                # Dimension
    n = 11               # Mode size
    m = int(1.E+4)       # Number of requests to the objective function
    f = func_build(d, n) # Target function, which defines the array elements

    t = tpc()
    i_opt, y_opt = protes(f, d, n, m, log=True)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}')


if __name__ == '__main__':
    demo()
