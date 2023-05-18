import numpy as np
from time import perf_counter as tpc


class Opti:
    def __init__(self, name='opti', with_arg_list=False, log=False):
        self.name = name
        self.with_arg_list = with_arg_list
        self.log = log

        self.err = ''
        self.is_prep = False
        self.is_done = False

        self.t = 0.

        self.m = 0
        self.i = None
        self.y = None
        self.e = None

        self.m_list = []
        self.i_list = []
        self.y_list = []
        self.e_list = []

        self.opts()

    def check(self, I, y):
        ind_opt = np.argmax(y) if self.is_max else np.argmin(y)

        i_cur = I[ind_opt, :]
        y_cur = y[ind_opt]

        is_new = self.y is None
        is_new = is_new or self.is_max and self.y < y_cur
        is_new = is_new or not self.is_max and self.y > y_cur

        if is_new:
            self.i = i_cur.copy()
            self.y = y_cur

            self.m_list.append(self.m)
            self.y_list.append(self.y)

            if self.y_real is not None:
                self.e = np.abs(self.y - self.y_real)
                self.e_list.append(self.e)

            if self.with_arg_list:
                self.i_list.append(self.i.copy())

            self.t = tpc() - self.t_start

            if self.log:
                print(self.info())

    def f(self, i, with_check=True):
        if self.is_f_batch:
            y = self.f_batch(i.reshape(1, -1), with_check=False)[0]
        else:
            y = self.f_(i)
            self.m += 1

        if with_check:
            self.check(i.reshape(1, -1), np.array([y]))

        return y

    def f_batch(self, I, with_check=True):
        if self.is_f_batch:
            y = self.f_(I)
            self.m += len(I)
        else:
            y = np.array([self.f(i, with_check=False) for i in I])

        if with_check:
            self.check(I, y)

        return y

    def info(self, len_name=12):
        name = self.name + ' '*max(0, len_name-len(self.name))
        text = f'{name} > '

        text += f'm {self.m:-7.1e} | '

        text += f't {self.t:-9.3e} | '

        if self.e is not None:
            text += f'e {self.e:-7.1e}'
        else:
            text += f'y {self.y:-11.5e}'

        if self.is_done:
            text += ' <<< DONE'

        return text

    def optimize(self):
        if not self.is_prep:
            self.err = 'Call "prep" method before usage'
        else:
            self._init()

        if self.err:
            raise ValueError(f'Method {self.name} is not ready ({self.err})')

        self.t_start = tpc()
        self._optimize()
        self.is_done = True
        self.t = tpc() - self.t_start
        if self.log:
            print(self.info())

    def opts(self):
        return

    def prep(self, f, d, n, m, y_real=None, is_max=False, is_f_batch=False):
        self.f_ = f
        self.d = d
        self.n = n
        self.m_max = int(m)
        self.y_real = y_real
        self.is_max = is_max
        self.is_f_batch = is_f_batch
        self.is_prep = True
        return self

    def _init(self):
        return

    def _optimize(self):
        raise NotImplementedError()

    def _optimize_ng(self, solver):
        import nevergrad as ng

        optimizer = solver(
            parametrization=ng.p.TransitionChoice(range(self.n[0]),
            repetitions=len(self.n)),
            budget=self.m_max,
            num_workers=1)

        recommendation = optimizer.provide_recommendation()

        for _ in range(optimizer.budget):
            x = optimizer.ask()
            i = np.array(x.value, dtype=int)
            optimizer.tell(x, self.f(i))
