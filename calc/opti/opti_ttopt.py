from opti import Opti
import numpy as np


try:
    from ttopt import TTOpt
    with_ttopt = True
except Exception as e:
    with_ttopt = False


class OptiTTOpt(Opti):
    def __init__(self, name='ttopt', *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def opts(self, with_qtt=True):
        self.opts_with_qtt = with_qtt

    def _init(self):
        if not with_ttopt:
            self.err = 'Need "ttopt" module'
            return

    def _optimize(self):
        if self.opts_with_qtt and self.n[0] != 2:
            # QTT-solver:
            n = None
            p = 2
            q = int(np.log2(self.n[0]))
            if p**q != self.n[0]:
                raise ValueError('Grid should be power of 2 for QTT')
        else:
            # TT-solver or binary tensor:
            n = self.n
            p = None
            q = None

        tto = TTOpt(self.f_batch, d=self.d, n=n, p=p, q=q, evals=self.m_max,
            is_func=False, is_vect=True)
        (tto.maximize if self.is_max else tto.minimize)()
