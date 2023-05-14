from opti import Opti


try:
    from protes import protes
    with_protes = True
except Exception as e:
    with_protes = False


class OptiProtes(Opti):
    def __init__(self, name='protes', *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def opts(self, k=50, k_top=5, k_gd=100, lr=1.E-4, r=5):
        self.opts_k = k
        self.opts_k_top = k_top
        self.opts_k_gd = k_gd
        self.opts_lr = lr
        self.opts_r = r

    def _init(self):
        if not with_protes:
            self.err = 'Need "protes" module'
            return

    def _optimize(self):
        protes(self.f_batch, self.d, self.n[0], self.m_max,
            k=self.opts_k, k_top=self.opts_k_top, k_gd=self.opts_k_gd,
            lr=self.opts_lr, r=self.opts_r, is_max=self.is_max)
