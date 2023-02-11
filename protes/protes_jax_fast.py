import jax
import jax.numpy as np
import optax
from time import perf_counter as tpc


def protes_jax_fast(f, n, m, k=50, k_top=5, k_gd=100, lr=1.E-4, r=5, P=None, seed=42, info={}, i_ref=None, is_max=False, log=False, log_ind=False, mod='jax', device='cpu'):
    time = tpc()
    info.update({'mod': mod, 'is_max': is_max, 'm': 0, 't': 0,
        'i_opt': None, 'y_opt': None, 'm_opt_list': [], 'y_opt_list': [],
        'm_ref_list': [],
        'p_ref_list': [], 'p_opt_ref_list': [], 'p_top_ref_list': []})

    rng = jax.random.PRNGKey(seed)

    if len(n) < 3 or len(set(n)) > 1:
        raise ValueError('It works only for constant mode size  3D+ tensors')

    if P is None:
        rng, key = jax.random.split(rng)
        P = _generate_initial(len(n), n[0], r, key)
    elif len(P[1].shape) != 4:
        raise ValueError('Initial P tensor should have special format')

    optim = optax.adam(lr)
    state = optim.init(P)

    interface_matrices = jax.jit(_interface_matrices)
    sample = jax.jit(jax.vmap(_sample, (None, None, None, None, 0)))
    likelihood = jax.jit(jax.vmap(_likelihood, (None, None, None, None, 0)))

    @jax.jit
    def loss(P_cur, I_cur):
        Pl, Pm, Pr = P_cur
        Zm = interface_matrices(Pm, Pr)
        l = likelihood(Pl, Pm, Pr, Zm, I_cur)
        return np.mean(-l)

    loss_grad = jax.grad(loss)

    @jax.jit
    def optimize(P, I_cur, state):
        grads = loss_grad(P, I_cur)
        updates, state = optim.update(grads, state)
        P = jax.tree_util.tree_map(lambda u, p: p + u, updates, P)
        return P, state

    while True:
        rng, key = jax.random.split(rng)
        Pl, Pm, Pr = P
        Zm = interface_matrices(Pm, Pr)
        I = sample(Pl, Pm, Pr, Zm, jax.random.split(key, k))

        y = f(I)
        y = np.array(y)
        info['m'] += y.shape[0]

        is_new = _check(I, y, info)

        if info['m'] >= m:
            break

        ind = np.argsort(y, kind='stable')
        ind = (ind[::-1] if is_max else ind)[:k_top]

        for _ in range(k_gd):
            P, state = optimize(P, I[ind, :], state)

        if i_ref is not None: # For debug only
            _set_ref(P, info, I, ind, i_ref)

        info['t'] = tpc() - time

        _log(info, log, log_ind, is_new)

    _log(info, log, log_ind, is_new, is_end=True)

    return info['i_opt'], info['y_opt']


def _check(I, y, info):
    """Check the current batch of function values and save the improvement."""
    ind_opt = np.argmax(y) if info['is_max'] else np.argmin(y)

    i_opt_curr = I[ind_opt, :]
    y_opt_curr = y[ind_opt]

    is_new = info['y_opt'] is None
    is_new = is_new or info['is_max'] and info['y_opt'] < y_opt_curr
    is_new = is_new or not info['is_max'] and info['y_opt'] > y_opt_curr

    if is_new:
        info['i_opt'] = i_opt_curr
        info['y_opt'] = y_opt_curr
        info['m_opt_list'].append(info['m'])
        info['y_opt_list'].append(y_opt_curr)
        return True


def _generate_initial(d, n, r, key):
    """Build initial random TT-tensor for probability."""
    keyl, keym, keyr = jax.random.split(key, 3)

    Yl = jax.random.uniform(keyl, (1, n, r))
    Ym = jax.random.uniform(keym, (d-2, r, n, r))
    Yr = jax.random.uniform(keyr, (r, n, 1))

    return [Yl, Ym, Yr]


def _get(Yl, Ym, Yr, i):
    """Compute the element of the TT-tensor Y for given multi-index i."""
    def scan(Q, data):
        I_cur, Y_cur = data

        Q = np.einsum('r,rq->q', Q, Y_cur[:, I_cur, :])

        return Q, None

    Q, _ = scan(np.ones(1), (i[0], Yl))
    Q, _ = jax.lax.scan(scan, Q, (i[1:-1], Ym))
    Q, _ = scan(Q, (i[-1], Yr))

    return Q[0]


def _interface_matrices(Ym, Yr):
    """Compute the "interface matrices" for the TT-tensor Y."""
    def scan(Z_prev, Y_cur):
        Z = np.sum(Y_cur, axis=1) @ Z_prev
        Z /= np.linalg.norm(Z)
        return Z, Z

    Q, Zr = scan(np.ones(1), Yr)
    Q, Zm = jax.lax.scan(scan, Q, Ym, reverse=True)

    return np.vstack((Zm[1:], Zr))


def _likelihood(Yl, Ym, Yr, Zm, i):
    """Compute the likelihood in a multi-index i for TT-tensor Y."""
    def scan(Q, data):
        I_cur, Y_cur, Z_cur = data

        G = np.einsum('r,riq,q->i', Q, Y_cur, Z_cur)
        G = np.abs(G)
        G /= np.sum(G)

        Q = np.einsum('r,rq->q', Q, Y_cur[:, I_cur, :])
        Q /= np.linalg.norm(Q)

        return Q, G[I_cur]

    Q, yl = scan(np.ones(1), (i[0], Yl, Yl[0, i[0], :]))
    Q, ym = jax.lax.scan(scan, Q, (i[1:-1], Ym, Zm))
    Q, yr = scan(Q, (i[-1], Yr, np.ones(1)))

    y = np.hstack((np.array(yl), ym, np.array(yr)))
    return np.sum(np.log(np.array(y)))


def _log(info, log=False, log_ind=False, is_new=False, is_end=False):
    """Print current optimization result to output."""
    if not log or (not is_new and not is_end):
        return

    text = f'protes-{info["mod"]} > '
    text += f'm {info["m"]:-7.1e} | '
    text += f't {info["t"]:-9.3e} | '
    text += f'y {info["y_opt"]:-11.4e}'

    if len(info["p_ref_list"]) > 0:
        text += f' | p_ref {info["p_ref_list"][-1]:-11.4e} | '

    if log_ind:
        text += f' | i {"".join([str(i) for i in info["i_opt"]])}'

    if is_end:
        text += ' <<< DONE'

    print(text)


def _sample(Yl, Ym, Yr, Zm, key):
    """Generate sample according to given probability TT-tensor Y."""
    def scan(Q, data):
        key_cur, Y_cur, Z_cur = data

        G = np.einsum('r,riq,q->i', Q, Y_cur, Z_cur)
        G = np.abs(G)
        G /= np.sum(G)

        i = jax.random.choice(key_cur, np.arange(Y_cur.shape[1]), p=G)

        Q = np.einsum('r,rq->q', Q, Y_cur[:, i, :])
        Q /= np.linalg.norm(Q)

        return Q, i

    keys = jax.random.split(key, len(Ym) + 2)

    Q, il = scan(np.ones(1), (keys[0], Yl, Zm[0]))
    Q, im = jax.lax.scan(scan, Q, (keys[1:-1], Ym, Zm))
    Q, ir = scan(Q, (keys[-1], Yr, np.ones(1)))

    il = np.array(il, dtype=np.int32)
    ir = np.array(ir, dtype=np.int32)
    return np.hstack((il, im, ir))


def _set_ref(P, info, I, ind, i_ref=None):
    info['m_ref_list'].append(info['m'])
    info['p_opt_ref_list'].append(_get(*P, info['i_opt']))
    info['p_top_ref_list'].append(_get(*P, I[ind[0], :]))
    if i_ref is not None:
        info['p_ref_list'].append(_get(*P, i_ref))
