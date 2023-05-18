import numpy as np
from opti import Opti


try:
    import teneva
    with_teneva = True
except Exception as e:
    with_teneva = False


class OptiOptimatt(Opti):
    def __init__(self, name='optimatt', *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def opts(self, dr_max=2):
        self.opts_dr_max = dr_max

    def _init(self):
        if not with_teneva:
            self.err = 'Need "teneva" module'
            return

    def _optimize(self):
        Y = teneva.rand(self.n, r=1)
        Y = teneva.cross(self.f_batch, Y, e=1.E-16, m=self.m_max,
            dr_max=self.opts_dr_max)
        Y = teneva.truncate(Y, e=1.E-16)

        i_min, y_min, i_max, y_max = teneva.optima_tt(Y)
        self.f(i_max if self.is_max else i_min)
