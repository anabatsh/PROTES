from opti import Opti


try:
    from protes import protes
    with_protes = True
except Exception as e:
    with_protes = False


class OptiProtes(Opti):
    def __init__(self, name='protes', *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def opts(self, k=100, k_top=10, k_gd=1, lr=5.E-2, r=5, P=None,
             seed=0):
        self.opts_k = k
        self.opts_k_top = k_top
        self.opts_k_gd = k_gd
        self.opts_lr = lr
        self.opts_r = r
        self.opts_P = P
        self.opts_seed = seed

    def _init(self):
        if not with_protes:
            self.err = 'Need "protes" module'
            return

    def _optimize(self):
        protes(self.f_batch, self.d, self.n[0], self.m_max, P=self.opts_P,
            k=self.opts_k, k_top=self.opts_k_top, k_gd=self.opts_k_gd,
            lr=self.opts_lr, r=self.opts_r, is_max=self.is_max,
            seed=self.opts_seed)
