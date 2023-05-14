from jax.config import config
config.update('jax_enable_x64', True)


import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


import numpy as np
from time import perf_counter as tpc


from teneva_bm import *
bms = [
    BmFuncAckley(d=7, n=16).prep(),
    BmFuncAlpine(d=7, n=16).prep(),
    BmFuncExp(d=7, n=16).prep(),
    BmFuncGriewank(d=7, n=16).prep(),
    BmFuncMichalewicz(d=7, n=16).prep(),
    BmFuncPiston(d=7, n=16).prep(),
    BmFuncQing(d=7, n=16).prep(),
    BmFuncRastrigin(d=7, n=16).prep(),
    BmFuncSchaffer(d=7, n=16).prep(),
    BmFuncSchwefel(d=7, n=16).prep(),

    BmQuboKnapAmba(d=50).prep(),
    BmQuboKnapQuad(d=50).prep(),
    BmQuboMaxcut(d=50).prep(),
    BmQuboMvc(d=50).prep(),

    BmOcSimple(d=25).prep(),
    BmOcSimple(d=50).prep(),
    BmOcSimple(d=100).prep(),

    BmOcSimpleConstr(d=25).prep(),
    BmOcSimpleConstr(d=50).prep(),
    BmOcSimpleConstr(d=100).prep(),
]


from opti import *
Optis = [
    OptiProtes,
    OptiTTOpt,
    OptiOptimatt,
    OptiOPO,
    OptiPSO,
    OptiNB,
    OptiSPSA,
    OptiPortfolio,
]


class Log:
    def __init__(self, fpath='log.txt'):
        self.fpath = fpath
        self.is_new = True
        self.len_pref = 19

        if os.path.dirname(self.fpath):
            os.makedirs(os.path.dirname(self.fpath), exist_ok=True)

    def __call__(self, text):
        print(text)
        with open(self.fpath, 'w' if self.is_new else 'a') as f:
            f.write(text + '\n')
        self.is_new = False


m = int(1.E+4)


log = Log('logs/result.txt')


for bm in bms:
    log(bm.info())
    for Opti in Optis:
        opti = Opti(log=False)
        opti.prep(bm.get, bm.d, bm.n, m, None, is_f_batch=True) # bm.y_min_real
        opti.optimize()
        log(opti.info())
    log('\n\n')
