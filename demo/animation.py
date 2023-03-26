import jax.numpy as jnp
import matplotlib as mpl
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from time import perf_counter as tpc


from protes import protes_general


mpl.rc('animation', html='jshtml')
mpl.rcParams['animation.embed_limit'] = 2**128


def _func_on_grid(f, n1, n2):
    I1 = np.arange(n1)
    I2 = np.arange(n2)

    I1, I2 = np.meshgrid(I1, I2)
    I = np.hstack([I1.reshape(-1, 1), I2.reshape(-1, 1)])
    Y = f(I).reshape(n1, n2)

    X1 = I1 / (n1 - 1) * (f.b - f.a) + f.a
    X2 = I2 / (n2 - 1) * (f.b - f.a) + f.a

    return X1, X2, Y


def _p_full(P):
    return np.einsum('riq, qjs->rijs', *P)[0, :, :, 0]


def _plot_2d(fig, ax, Y, i_opt_real=None):
    img = ax.imshow(Y, cmap=cm.coolwarm, alpha=0.8)
    if i_opt_real is not None:
        ax.scatter(*i_opt_real, s=500, c='#ffbf00', marker='*', alpha=0.9)
    ax.set_xlim(0, Y.shape[0])
    ax.set_ylim(0, Y.shape[1])
    ax.axis('off')
    return img


def _plot_3d(fig, ax, title, X1, X2, Y):
    ax.set_title(title, fontsize=16)
    surf = ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # fig.colorbar(surf, ax=ax, shrink=0.3, aspect=10)
    return surf


def animate(f, n1, n2, info, i_opt_real, fpath=None):
    y_opt_real = f(i_opt_real.reshape(1, -1))[0]

    fig = plt.figure(figsize=(16, 16))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)

    X1, X2, Y = _func_on_grid(f, n1, n2)
    P = _p_full(info['P_list'][0])

    img_y_3d = _plot_3d(fig, ax1, 'Target function', X1, X2, Y)
    img_p_3d = _plot_3d(fig, ax3, 'Probability tensor', X1, X2, P)

    img_y_2d = _plot_2d(fig, ax2, Y, i_opt_real)
    img_p_2d = _plot_2d(fig, ax4, P, i_opt_real)

    img_opt = ax2.scatter(0, 0, s=150, c='#EE17DA', marker='D')
    img_req = ax2.scatter(0, 0, s= 70, c='#8b1d1d')
    img_req_top1 = ax2.scatter(0, 0, s= 110, c='#ffcc00', alpha=0.8)
    img_req_top2 = ax4.scatter(0, 0, s= 110, c='#ffcc00')
    img_hist, = ax2.plot([], [], '--', c='#485536', linewidth=1, markersize=0)

    def update(k, *args):
        i_opt = info['i_opt_list'][k]
        y_opt = info['y_opt_list'][k]
        m = info['m_opt_list'][k]
        I = info['I_list'][k]
        y = info['y_list'][k]
        e = abs(y_opt_real - y_opt)
        P = _p_full(info['P_list'][k])

        ind = jnp.argsort(y, kind='stable')
        ind = (ind[::-1] if info['is_max'] else ind)[:info['k_top']]
        I_top = I[ind, :]

        ax3.clear()
        _plot_3d(fig, ax3, 'Probability tensor', X1, X2, P)

        img_p_2d.set_array(P)

        img_opt.set_offsets(np.array([i_opt[0], i_opt[1]]))
        img_req.set_offsets(I)
        img_req_top1.set_offsets(I_top)
        img_req_top2.set_offsets(I_top)

        pois_x, pois_y = [], []
        for i in info['i_opt_list'][:(k+1)]:
            pois_x.append(i[0])
            pois_y.append(i[1])
        img_hist.set_data(pois_x, pois_y)

        ax2.set_title(f'Queries: {m:-7.1e} | Error : {e:-7.1e}', fontsize=20)

        return img_p_2d, img_opt, img_req, img_req_top1, img_req_top2, img_hist

    anim = FuncAnimation(fig, update, interval=30,
        frames=len(info['y_list']), blit=True, repeat=False)

    if fpath:
        anim.save(fpath, writer='pillow', fps=0.7)
    else:
        anim.show()


def run(d=2, n=501, m=int(1.E+4), k=50, k_top=5):
    """Animation of the PROTES work for the 2D case."""
    def func_build(d, n):
        """Ackley function. See https://www.sfu.ca/~ssurjano/ackley.html."""
        a = -32.768         # Grid lower bound
        b = +32.768         # Grid upper bound

        par_a = 20.         # Standard parameter values for Ackley function
        par_b = 0.2
        par_c = 2.*jnp.pi

        def func(I):
            """Target function: y=f(I); [samples,d] -> [samples]."""
            n_ext = np.repeat(n.reshape((1, -1)), I.shape[0], axis=0)
            X = I / (n_ext - 1) * (b - a) + a

            y1 = jnp.sqrt(jnp.sum(X**2, axis=1) / d)
            y1 = - par_a * jnp.exp(-par_b * y1)

            y2 = jnp.sum(jnp.cos(par_c * X), axis=1)
            y2 = - jnp.exp(y2 / d)

            y3 = par_a + jnp.exp(1.)

            return y1 + y2 + y3

        func.a = a
        func.b = b
        return func

    t =tpc()
    print('\n... start optimization ...')
    info = {}
    f = func_build(d, jnp.array([n, n]))
    i_opt, y_opt = protes_general(f, [n, n], m, k, k_top, log=True,
        info=info, with_info_full=True)
    print(f'Optimization is ready (total time {tpc()-t:-8.2f} sec)')

    i_opt_real = np.array([int(n/2), int(n/2)])

    t =tpc()
    print('\n... start building animation ...')
    animate(f, n, n, info, i_opt_real, fpath='./demo/animation.gif')
    print(f'Animation is ready (total time {tpc()-t:-8.2f} sec)')


if __name__ == '__main__':
    run()
