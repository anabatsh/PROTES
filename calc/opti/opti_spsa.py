import numpy as np
from opti import Opti


try:
    import nevergrad as ng
    with_ng = True
except Exception as e:
    with_ng = False


class OptiSPSA(Opti):
    def __init__(self, name='spsa', *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def _init(self):
        if not with_ng:
            self.err = 'Need "nevergrad" module'
            return

    def _optimize(self):
        self._optimize_ng(ng.optimizers.SPSA)
