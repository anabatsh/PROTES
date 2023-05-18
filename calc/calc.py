import matplotlib as mpl
import numpy as np
import os
import pickle
import sys
from time import perf_counter as tpc


mpl.rcParams.update({
    'font.family': 'normal',
    'font.serif': [],
    'font.sans-serif': [],
    'font.monospace': [],
    'font.size': 12,
    'text.usetex': False,
})


import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


sns.set_context('paper', font_scale=2.5)
sns.set_style('white')
sns.mpl.rcParams['legend.frameon'] = 'False'


from jax.config import config
config.update('jax_enable_x64', True)
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


import jax.numpy as jnp


from constr import ind_tens_max_ones


from teneva_bm import *
bms = [
    BmFuncAckley(d=7, n=16, name='P-01'),
    BmFuncAlpine(d=7, n=16, name='P-02'),
    BmFuncExp(d=7, n=16, name='P-03'),
    BmFuncGriewank(d=7, n=16, name='P-04'),
    BmFuncMichalewicz(d=7, n=16, name='P-05'),
    BmFuncPiston(d=7, n=16, name='P-06'),
    BmFuncQing(d=7, n=16, name='P-07'),
    BmFuncRastrigin(d=7, n=16, name='P-08'),
    BmFuncSchaffer(d=7, n=16, name='P-09'),
    BmFuncSchwefel(d=7, n=16, name='P-10'),

    BmQuboMaxcut(d=50, name='P-11'),
    BmQuboMvc(d=50, name='P-12'),
    BmQuboKnapQuad(d=50, name='P-13'),
    BmQuboKnapAmba(d=50, name='P-14'),

    BmOcSimple(d=25, name='P-15'),
    BmOcSimple(d=50, name='P-16'),
    BmOcSimple(d=100, name='P-17'),

    BmOcSimpleConstr(d=25, name='P-18'),
    BmOcSimpleConstr(d=50, name='P-19'),
    BmOcSimpleConstr(d=100, name='P-20'),
]


BM_FUNC      = ['P-01', 'P-02', 'P-03', 'P-04', 'P-05', 'P-06', 'P-07',
                'P-08', 'P-09', 'P-10']
BM_QUBO      = ['P-11', 'P-12', 'P-13', 'P-14']
BM_OC        = ['P-15', 'P-16', 'P-17']
BM_OC_CONSTR = ['P-18', 'P-19', 'P-20']


from opti import *
Optis = {
    'Our': OptiProtes,
    'BS-1': OptiTTOpt,
    'BS-2': OptiOptimatt,
    'BS-3': OptiOPO,
    'BS-4': OptiPSO,
    'BS-5': OptiNB,
    'BS-6': OptiSPSA,
    'BS-7': OptiPortfolio,
}


class Log:
    def __init__(self, fpath='log.txt'):
        self.fpath = fpath
        self.is_new = True

        if os.path.dirname(self.fpath):
            os.makedirs(os.path.dirname(self.fpath), exist_ok=True)

    def __call__(self, text):
        print(text)
        with open(self.fpath, 'w' if self.is_new else 'a') as f:
            f.write(text + '\n')
        self.is_new = False


def calc(m=int(1.E+4), seed=0):
    log = Log()
    res = {}

    for bm in bms:
        np.random.seed(seed)
        if bm.name in BM_FUNC:
            # We carry out a small random shift of the function's domain,
            # so that the optimum does not fall into the middle of the domain:
            bm = _prep_bm_func(bm)
        else:
            bm.prep()

        log(bm.info())
        res[bm.name] = {}

        for opti_name, Opti in Optis.items():
            np.random.seed(seed)
            opti = Opti(name=opti_name)
            opti.prep(bm.get, bm.d, bm.n, m, is_f_batch=True)

            if bm.name in BM_OC_CONSTR and opti_name == 'Our':
                # Problem with constraint for PROTES (we use the initial
                # approximation of the special form in this case):
                P = ind_tens_max_ones(bm.d, 3, opti.opts_r)
                Pl = jnp.array(P[0], copy=True)
                Pm = jnp.array(P[1:-1], copy=True)
                Pr = jnp.array(P[-1], copy=True)
                P = [Pl, Pm, Pr]
                opti.opts(P=P)

            opti.optimize()

            log(opti.info())
            res[bm.name][opti.name] = [opti.m_list, opti.y_list, opti.y]
            _save(res)

        log('\n\n')


def plot(m_min=1.E+0):
    plot_opts = {
        'P-02': {},
        'P-14': {'y_min': 1.8E+3, 'y_max': 3.2E+3, 'inv': True},
        'P-16': {'y_min': 1.E-2, 'y_max': 2.E+0},
    }
    res = _load()

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    plt.subplots_adjust(wspace=0.3)

    i = -1
    for bm, item in res.items():
        if not bm in plot_opts.keys():
            continue
        i += 1
        ax = axs[i]

        ax.set_xlabel('Number of requests')

        for opti, data in item.items():
            m = np.array(data[0], dtype=int)
            y = np.array(data[1])
            if plot_opts[bm].get('inv'):
                y *= -1
            j = np.argmax(m >= m_min)
            nm = opti
            if nm == 'Our':
                nm = 'PROTES'
            ax.plot(m[j:], y[j:], label=nm,
                marker='o', markersize=8, linewidth=6 if nm == 'PROTES' else 3)

        _prep_ax(ax, xlog=True, ylog=True, leg=i==0)
        ax.set_xlim(m_min, 2.E+4)
        if 'y_min' in plot_opts[bm]:
            ax.set_ylim(plot_opts[bm]['y_min'], plot_opts[bm]['y_max'])

    #yticks = [1.8E+3, 2.0E+3, 2.2E+3, 2.4E+3, 2.6E+3, 2.8E+3, 3.0E+3, 3.2E+3]
    #ax.set(yticks=yticks, yticklabels=[int(])
    #ax.get_yaxis().get_major_formatter().labelOnlyBase = False
    plt.savefig('deps.png', bbox_inches='tight')


def text():
    res = _load()

    text =  '\n\n% ' + '='*50 + '\n' + '% [START] Auto generated data \n\n'

    for i, (bm, item) in enumerate(res.items(), 1):
        if i in [11, 15, 18]:
            text += '\n\\hline\n'
        if i == 1:
            text += '\\multirow{10}{*}{\\parbox{1.6cm}{Analytic Functions}}\n'
        if i == 11:
            text += '\\multirow{3}{*}{QUBO}\n'
        if i == 15:
            text += '\\multirow{3}{*}{Control}\n'
        if i == 18:
            text += '\\multirow{3}{*}{\parbox{1.67cm}{Control +constr.}}\n'

        text += f'    & {bm}\n'
        vals = np.array([v[2] for v in item.values()])
        for v in vals:
            if v < 1.E+40:
                text += f'        & {v:-8.1e}\n'
            else:
                text += f'        & Fail\n'
        text += f'    \\\\ \n'
    text += '\n\n\\hline\n\n'
    text += '\n% [END] Auto generated data \n% ' + '='*50 + '\n\n'
    print(text)


def _load(fpath='res.pickle'):
    with open(fpath, 'rb') as f:
        res = pickle.load(f)
    return res


def _prep_ax(ax, xlog=False, ylog=False, leg=False, xint=False, xticks=None):
    if xlog:
        ax.semilogx()
    if ylog:
        ax.semilogy()

    if leg:
        ax.legend(loc='upper right', frameon=True)

    ax.grid(ls=":")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if xint:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if xticks is not None:
        ax.set(xticks=xticks, xticklabels=xticks)


def _prep_bm_func(bm):
    shift = np.random.randn(bm.d) / 10
    a_new = bm.a - (bm.b-bm.a) * shift
    b_new = bm.b + (bm.b-bm.a) * shift
    bm.set_grid(a_new, b_new)
    bm.prep()
    return bm


def _save(res, fpath='res.pickle'):
    with open(fpath, 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'calc'

    if mode == 'calc':
        calc()
    elif mode == 'plot':
        plot()
    elif mode == 'text':
        text()
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
