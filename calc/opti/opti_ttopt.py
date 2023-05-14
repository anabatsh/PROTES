from opti import Opti


try:
    from ttopt import TTOpt
    with_ttopt = True
except Exception as e:
    with_ttopt = False


class OptiTTOpt(Opti):
    def __init__(self, name='ttopt', *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def opts(self, r=5, p=None):
        self.opts_r = r
        self.opts_p = p

    def _init(self):
        if not with_ttopt:
            self.err = 'Need "ttopt" module'
            return

    def _optimize(self):
        tto = TTOpt(self.f_batch, d=self.d, n=self.n, evals=self.m_max,
            is_func=False, is_vect=True)
        func = tto.maximize if self.is_max else tto.minimize
        func(self.opts_r)
