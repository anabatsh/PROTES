from time import perf_counter as tpc
import torch


def protes_tor(f, n, m, k=50, k_top=5, k_gd=100, lr=1.E-4, r=5, P=None, seed=42, info={}, i_ref=None, is_max=False, log=False, log_ind=False, mod='tor', device='cpu'):
    time = tpc()
    info.update({'mod': mod, 'is_max': is_max, 'm': 0, 't': 0,
        'i_opt': None, 'y_opt': None, 'm_opt_list': [], 'y_opt_list': [],
        'm_ref_list': [],
        'p_ref_list': [], 'p_opt_ref_list': [], 'p_top_ref_list': []})

    torch.manual_seed(seed)

    if P is None:
        P = _generate_initial(n, r)
    P = [G.clone().detach().to(device).requires_grad_(True) for G in P]

    optim = torch.optim.Adam(P, lr)

    def loss(P_cur, I_cur):
        return torch.mean(-_likelihood_many(P_cur, I_cur))

    def optimize(P, I_cur):
        optim.zero_grad()
        l = loss(P, I_cur)
        l.backward()
        optim.step()
        return P

    while True:
        with torch.no_grad():
            I = _sample_many(P, k)

        y = f(I)
        y = torch.tensor(y, device=device)
        info['m'] += y.shape[0]

        is_new = _check(I, y, info)

        if info['m'] >= m:
            break

        ind = torch.argsort(y)
        ind = (ind[::-1] if is_max else ind)[:k_top]

        for _ in range(k_gd):
            P = optimize(P, I[ind, :])

        if i_ref is not None: # For debug only
            with torch.no_grad():
                _set_ref(P, info, I, ind, i_ref)

        info['t'] = tpc() - time

        _log(info, log, log_ind, is_new)

    _log(info, log, log_ind, is_new, is_end=True)

    return info['i_opt'], info['y_opt']


def _check(I, y, info):
    """Check the current batch of function values and save the improvement."""
    ind_opt = torch.argmax(y) if info['is_max'] else torch.argmin(y)

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


def _generate_initial(n, r):
    """Build initial random TT-tensor for probability."""
    d = len(n)
    r = [1] + [r]*(d-1) + [1]

    Y = []
    for j in range(d):
        Y.append(torch.rand((r[j], n[j], r[j+1])))

    return Y


def _get(Y, i):
    """Compute the element of the TT-tensor Y for given multi-index i."""
    Q = Y[0][0, i[0], :]
    for j in range(1, len(Y)):
        Q = torch.einsum('r,rq->q', Q, Y[j][:, i[j], :])
    return Q[0]


def _interface_matrices(Y):
    """Compute the "interface matrices" for the TT-tensor Y."""
    d = len(Y)
    Z = [[]] * (d+1)
    Z[0] = torch.ones(1, dtype=Y[0].dtype, device=Y[0].device)
    Z[d] = torch.ones(1, dtype=Y[0].dtype, device=Y[0].device)
    for j in range(d-1, 0, -1):
        Z[j] = torch.sum(Y[j], dim=1) @ Z[j+1]
        Z[j] = Z[j] / torch.norm(Z[j])
    return Z


def _likelihood_many(Y, I):
    """Compute the likelihood in a multi-index I for TT-tensor Y."""
    d = len(Y)

    Z = _interface_matrices(Y)

    G = torch.einsum('riq,q->i', Y[0], Z[1])
    G = torch.abs(G)
    G = G / G.sum()

    y = [G[I[:, 0]]]

    Z[0] = Y[0][0, I[:, 0], :]

    for j in range(1, d):
        G = torch.einsum('jr,riq,q->ji', Z[j-1], Y[j], Z[j+1])
        G = torch.abs(G)
        G = G / G.sum(axis=1)[:, None]

        y.append(G[torch.arange(torch.numel(I[:, j])), I[:, j]])

        Z[j] = torch.einsum('ir,riq->iq', Z[j-1], Y[j][:, I[:, j], :])
        Z[j] = Z[j] / torch.linalg.norm(Z[j], axis=1)[:, None]

    return torch.sum(torch.log(torch.vstack(y).T), axis=1)


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


def _sample_many(Y, k):
    """Generate k samples according to given probability TT-tensor Y."""
    d = len(Y)
    I = torch.zeros((k, d), dtype=torch.long, device=Y[0].device)

    Z = _interface_matrices(Y)

    G = torch.einsum('riq,q->i', Y[0], Z[1])
    G = torch.abs(G)
    G = G / G.sum()

    i = torch.multinomial(G, k, replacement=True)
    I[:, 0] = i

    Z[0] = Y[0][0, i, :]

    for j in range(1, d):
        G = torch.einsum('jr,riq,q->ji', Z[j-1], Y[j], Z[j+1])
        G = torch.abs(G)
        G = G / G.sum(axis=1)[:, None]

        i = torch.tensor([
            torch.multinomial(g, 1, replacement=True) for g in G])
        I[:, j] = i

        Z[j] = torch.einsum("ir,riq->iq", Z[j-1], Y[j][:, i, :])
        Z[j] = Z[j] / torch.linalg.norm(Z[j], axis=1)[:, None]


    return I


def _set_ref(P, info, I, ind, i_ref=None):
    info['m_ref_list'].append(info['m'])
    info['p_opt_ref_list'].append(_get(P, info['i_opt']))
    info['p_top_ref_list'].append(_get(P, I[ind[0], :]))
    if i_ref is not None:
        info['p_ref_list'].append(_get(P, i_ref))
