import numpy as np
import os
import sys


from protes import animation


def func_build_ackley(n):
    """Ackley function. See https://www.sfu.ca/~ssurjano/ackley.html."""
    d = 2               # Dimension
    a = -32.768         # Grid lower bound
    b = +32.768         # Grid upper bound

    par_a = 20.         # Standard parameter values for Ackley function
    par_b = 0.2
    par_c = 2.*np.pi

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        n_ext = np.repeat(n.reshape((1, -1)), I.shape[0], axis=0)
        X = I / (n_ext - 1) * (b - a) + a

        y1 = np.sqrt(np.sum(X**2, axis=1) / d)
        y1 = - par_a * np.exp(-par_b * y1)

        y2 = np.sum(np.cos(par_c * X), axis=1)
        y2 = - np.exp(y2 / d)

        y3 = par_a + np.exp(1.)

        return y1 + y2 + y3

    i_opt_real = np.array([int(n[0]/2), int(n[1]/2)])
    return func, a, b, i_opt_real


def func_build_simple(n):
    d = 2           # Dimension
    a = -2.         # Grid lower bound
    b = +2.         # Grid upper bound

    i_opt_real = np.array([int(n[0] * 0.5), int(n[1] * 0.5)])
    x_opt_real = i_opt_real / (n - 1) * (b - a) + a

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        n_ext = np.repeat(n.reshape((1, -1)), I.shape[0], axis=0)
        X = I / (n_ext - 1) * (b - a) + a

        y0 = +0.
        y1 = -(X[:, 0] - x_opt_real[0])**2 - 2.0 * np.sin(X[:, 0])**2
        y2 = -(X[:, 1] - x_opt_real[1])**2 - 2.5 * np.sin(X[:, 1])**2
        return y0 + y1 + y2

    return func, a, b, i_opt_real


def func_build_two_optima(n, s=0.1, x0=4.):
    d = 2           # Dimension
    a = -5.         # Grid lower bound
    b = +5.         # Grid upper bound

    i_opt_real = None

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        n_ext = np.repeat(n.reshape((1, -1)), I.shape[0], axis=0)
        X = I / (n_ext - 1) * (b - a) + a
        y1 = np.exp(-s*np.sum((X - x0)**2, axis=1))
        y2 = np.exp(-s*np.sum((X + x0)**2, axis=1))
        return y1 + y2

    return func, a, b, i_opt_real


def animate(task):
    """Animation of the PROTES work for the 2D case."""
    fpath = os.path.dirname(__file__) + f'/protes_{task}.gif'

    if task == 'ackley':
        n = 101
        f, a, b, i_opt_real = func_build_ackley(np.array([n, n]))
        animation(f, a, b, n, m=int(2.E+2), k=25, k_top=5, k_gd=10, lr=1.E-2,
            i_opt_real=i_opt_real, fpath=fpath)

    elif task == 'simple':
        n = 101
        f, a, b, i_opt_real = func_build_simple(np.array([n, n]))
        animation(f, a, b, n, m=int(1.1E+3), k=100, k_top=10, k_gd=1, lr=5.E-2,
            i_opt_real=i_opt_real, fpath=fpath, is_max=True)

    elif task == 'two_optima':
        n = 101
        f, a, b, i_opt_real = func_build_two_optima(np.array([n, n]))
        animation(f, a, b, n, m=int(5.E+2), k=25, k_top=1, k_gd=1, lr=1.E-1,
            i_opt_real=i_opt_real, fpath=fpath, is_max=True)

    else:
        raise NotImplementedError(f'Task name "{task}" is not supported')


if __name__ == '__main__':
    task = sys.argv[1] if len(sys.argv) > 1 else 'simple'
    animate(task)
