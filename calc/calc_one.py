import numpy as np
import os
from time import perf_counter as tpc


from jax.config import config
config.update('jax_enable_x64', True)
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


from protes import protes
from teneva_bm import BmQuboKnapAmba


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
    def __init__(self, fpath='log_one.txt'):
        self.fpath = fpath
        self.is_new = True

        if os.path.dirname(self.fpath):
            os.makedirs(os.path.dirname(self.fpath), exist_ok=True)

    def __call__(self, text):
        print(text)
        with open(self.fpath, 'w' if self.is_new else 'a') as f:
            f.write(text + '\n')
        self.is_new = False


def calc_one(m=int(1.E+5), rep=10):
    log = Log()
    res = {}

    bm = BmQuboKnapAmba(d=50, name='P-14').prep()
    log(bm.info())

    for name, Opti in Optis.items():
        res[name] = []
        for seed in range(rep):
            np.random.seed(seed)
            opti = Opti(name=name)
            opti.prep(bm.get, bm.d, bm.n, m, is_f_batch=True)
            if name == 'Our':
                opti.opts(seed=seed)

            opti.optimize()
            res[name].append(opti.y)
            log(opti.info() + f' # {seed+1:-3d}')
        log('')

    text = '\n\n\n\n--- RESULT ---\n\n'
    for name, Opti in Optis.items():
        y = np.array(res[name])
        text += name + ' '*max(0, 10-len(name)) + ' >>> '
        text += f'Mean: {np.mean(y):-12.6e}  |  Best: {np.min(y):-12.6e}\n'
    log(text)


if __name__ == '__main__':
    calc_one()
