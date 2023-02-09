import jax
import jax.numpy as np
import optax
from time import perf_counter as tpc


def protes_jax(f, n, m, k=50, k_top=5, k_gd=100, lr=1.E-4, r=5, P=None, seed=42, info={}, i_ref=None, is_max=False, log=False, log_ind=False, mod='jax'):
    time = tpc()
    info.update({'mod': mod, 'is_max': is_max, 'm': 0, 't': 0,
        'i_opt': None, 'y_opt': None, 'm_opt_list': [], 'y_opt_list': [],
        'm_ref_list': [],
        'p_ref_list': [], 'p_opt_ref_list': [], 'p_top_ref_list': []})

    rng = jax.random.PRNGKey(seed)

    optim = optax.adam(lr)
    sample = jax.jit(jax.vmap(_sample, (None, 0)))
    likelihood = jax.jit(jax.vmap(_likelihood, (None, 0)))

    if P is None:
        rng, key = jax.random.split(rng)
        P = _generate_initial(n, r, key)

    state = optim.init(P)

    @jax.jit
    def loss(P_cur, I_cur):
        return np.mean(-likelihood(P_cur, I_cur))

    loss_grad = jax.grad(loss)

    @jax.jit
    def optimize(P, I_cur, state):
        grads = loss_grad(P, I_cur)
        updates, state = optim.update(grads, state)
        P = jax.tree_util.tree_map(lambda u, p: p + u, updates, P)
        return P, state

    while True:
        rng, key = jax.random.split(rng)
        I = sample(P, jax.random.split(key, k))

        y = f(I)
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


def _generate_initial(n, r, key):
    """Build initial random TT-tensor for probability."""
    d = len(n)
    r = [1] + [r]*(d-1) + [1]
    keys = jax.random.split(key, d)

    Y = []
    for j in range(d):
        Y.append(jax.random.uniform(keys[j], (r[j], n[j], r[j+1])))

    return Y


def _get(Y, i):
    """Compute the element of the TT-tensor Y for given multi-index i."""
    Q = Y[0][0, i[0], :]
    for j in range(1, len(Y)):
        Q = np.einsum('r,rq->q', Q, Y[j][:, i[j], :])
    return Q[0]


def _interface_matrices(Y):
    """Compute the "interface matrices" for the TT-tensor Y."""
    d = len(Y)
    Z = [[]] * (d+1)
    Z[0] = np.ones(1)
    Z[d] = np.ones(1)
    for j in range(d-1, 0, -1):
        Z[j] = np.sum(Y[j], axis=1) @ Z[j+1]
        Z[j] = Z[j] / np.linalg.norm(Z[j])
    return Z


def _likelihood(Y, I):
    """Compute the likelihood in a multi-index I for TT-tensor Y."""
    d = len(Y)

    Z = _interface_matrices(Y)

    G = np.einsum('riq,q->i', Y[0], Z[1])
    G = np.abs(G)
    G /= G.sum()

    y = [G[I[0]]]

    Z[0] = Y[0][0, I[0], :]

    for j in range(1, d):
        G = np.einsum('r,riq,q->i', Z[j-1], Y[j], Z[j+1])
        G = np.abs(G)
        G /= np.sum(G)

        y.append(G[I[j]])

        Z[j] = Z[j-1] @ Y[j][:, I[j], :]
        Z[j] /= np.linalg.norm(Z[j])

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


def _sample(Y, key):
    """Generate sample according to given probability TT-tensor Y."""
    d = len(Y)
    keys = jax.random.split(key, d)
    I = np.zeros(d, dtype=np.int32)

    Z = _interface_matrices(Y)

    G = np.einsum('riq,q->i', Y[0], Z[1])
    G = np.abs(G)
    G /= G.sum()

    i = jax.random.choice(keys[0], np.arange(Y[0].shape[1]), p=G)
    I = I.at[0].set(i)

    Z[0] = Y[0][0, I[0], :]

    for j in range(1, d):
        G = np.einsum('r,riq,q->i', Z[j-1], Y[j], Z[j+1])
        G = np.abs(G)
        G /= np.sum(G)

        i = jax.random.choice(keys[j], np.arange(Y[j].shape[1]), p=G)
        I = I.at[j].set(i)

        Z[j] = Z[j-1] @ Y[j][:, I[j], :]
        Z[j] /= np.linalg.norm(Z[j])

    return I


def _set_ref(P, info, I, ind, i_ref=None):
    info['m_ref_list'].append(info['m'])
    info['p_opt_ref_list'].append(_get(P, info['i_opt']))
    info['p_top_ref_list'].append(_get(P, I[ind[0], :]))
    if i_ref is not None:
        info['p_ref_list'].append(_get(P, i_ref))
