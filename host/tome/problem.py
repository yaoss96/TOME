"""The problem instance of the board measurements, and how a run is scored.

S is a dense row-wise softmax over a SuiteSparse graph: every node gets a
background logit that is constant along its row, its graph neighbours and
self-loop get larger logits (seed 0xBADC0FFE).  The initial frontier has
1-4 active states in each of M rows (seed 0x5A5A).  Every engine of the
paper was measured on exactly this instance; digest() fingerprints it.

Runs are scored against one exact chain, P <- TopK(P S) in FP64.  Because
the background logit is constant along a row, P S holds long runs of equal
scores, and which of them a Top-k keeps is an arbitrary tie-break, so the
paper reports the tie-aware score (score_tie): a retained state counts if
its exact score reaches the exact k-th largest score of its row.
"""
import hashlib

import numpy as np
from scipy import sparse
from scipy.io import mmread


def gen_frontier(M, N, mx_act=4, seed=0xA5):
    rng = np.random.RandomState(seed)
    nz = []
    for q in range(M):
        k = min(rng.randint(1, mx_act + 1), N)
        cols = rng.choice(N, size=k, replace=False)
        lg = rng.uniform(-1.0, 2.5, size=k)
        lg[0] = rng.uniform(0.8, 2.2)
        mx = lg.max()
        e = np.exp(lg - mx)
        p = e / e.sum()
        for i in range(k):
            nz.append((q, int(cols[i]), float(p[i])))
    return nz


def gen_attention_from_adj(adj_csr, N, seed=0xBADC0FFE):
    rng = np.random.RandomState(seed)
    S = np.full((N, N), -8.0, dtype=np.float64)
    for r in range(N):
        temp = rng.uniform(0.5, 2.0)
        bg = (-6.0 - 2.0 * rng.random()) / temp
        S[r, :] = bg
        S[r, r] = rng.uniform(1.5, 2.5) / temp
        s, e = adj_csr.indptr[r], adj_csr.indptr[r + 1]
        for p in range(s, e):
            nb = adj_csr.indices[p]
            if nb == r:
                continue
            roll = rng.random()
            if roll > 0.70:
                lg = rng.uniform(1.0, 2.5)
            elif roll > 0.30:
                lg = rng.uniform(-1.0, 1.0)
            else:
                lg = rng.uniform(-3.0, -1.5)
            S[r, nb] = lg / temp
        mx = S[r].max()
        ex = np.exp(S[r] - mx)
        S[r] = ex / ex.sum()
    return S.astype(np.float32)


def build(mtx, nmax=0, m_cap=256):
    """S and the initial frontier for a Matrix Market graph; nmax > 0 keeps
    the subgraph induced by the first nmax nodes."""
    adj = mmread(mtx).tocsr()
    n_graph = adj.shape[0]
    n = min(n_graph, nmax) if nmax else n_graph
    if n < n_graph:
        adj = adj[:n, :n]
    adj = sparse.csr_matrix(adj, shape=(n, n))
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj.eliminate_zeros()
    S = gen_attention_from_adj(adj, n)
    m = min(m_cap, n)
    nz = gen_frontier(m, n, 4, seed=0x5A5A)
    return dict(n_graph=n_graph, N=n, M=m, S=S, nz0=nz, adj_nnz=int(adj.nnz))


def nz_to_rows(nz, M):
    """[(r, c, v)] -> per row (cols ascending, vals)"""
    buf = [[] for _ in range(M)]
    for r, c, v in nz:
        if 0 <= r < M:
            buf[r].append((c, v))
    out = []
    for b in buf:
        b.sort()
        out.append((np.array([c for c, _ in b], dtype=np.int64),
                    np.array([v for _, v in b], dtype=np.float64)))
    return out


def rows_to_csr(rows, N):
    ptr = np.zeros(len(rows) + 1, np.int64)
    for i, (c, _) in enumerate(rows):
        ptr[i + 1] = ptr[i] + len(c)
    col = np.concatenate([c for c, _ in rows]) if ptr[-1] else np.zeros(0, np.int64)
    val = np.concatenate([v for _, v in rows]) if ptr[-1] else np.zeros(0)
    return sparse.csr_matrix((val, col, ptr), shape=(len(rows), N))


def topk_dense(Y, k):
    """per row: the k largest |entries| of a dense row, cols ascending"""
    out = []
    for i in range(Y.shape[0]):
        v = Y[i]
        nz = np.nonzero(v)[0]
        if len(nz) > k:
            nz = nz[np.argsort(-np.abs(v[nz]), kind='stable')[:k]]
        o = np.sort(nz)
        out.append((o, v[o]))
    return out


class Hop(list):
    """One hop of the exact chain: its Top-k rows, plus the dense FP64
    product Y they were selected from (for the tie-aware score)."""
    Y = None


def exact_chain(inst, k, hops):
    """FP64 P <- TopK(P S); one Hop per hop."""
    S = inst['S'].astype(np.float64)
    M, N = inst['M'], inst['N']
    P = rows_to_csr(nz_to_rows(inst['nz0'], M), N)
    out = []
    for _ in range(hops):
        Y = np.asarray(P @ S)
        rows = Hop(topk_dense(Y, k))
        rows.Y = Y
        out.append(rows)
        P = rows_to_csr(rows, N)
    return out


def score_tie(ex, q_rows, k, rtol=1e-9):
    """Tie-aware fidelity against the exact chain.

    S's background logit is constant along each of its rows, so a row of
    P S holds long runs of exactly equal scores, and which of them a
    Top-k keeps is an arbitrary tie-break.  The plain overlap (score)
    counts such a choice as a miss.  Here a retained state counts as correct if
    its exact score reaches the exact k-th largest score of its row, and
    mass is the exact score of the retained states over the best
    attainable.
    """
    ov, mr = [], []
    for i, (qc, qv) in enumerate(q_rows):
        y = np.abs(ex.Y[i])
        kk = min(k, int(np.count_nonzero(y)))
        if kk == 0:
            continue
        top = np.partition(y, len(y) - kk)[len(y) - kk:]
        tau = top.min()
        qc = np.asarray(qc)[np.abs(np.asarray(qv)) > 0][:kk]
        hit = int((y[qc] >= tau * (1 - rtol)).sum())
        ov.append(hit / kk)
        mr.append(float(y[qc].sum() / top.sum()))
    return float(np.mean(ov)), float(np.mean(mr))


def score(ex_rows, q_rows, k):
    """Plain Top-k overlap and retained-mass recall (sim/sweep.py's metrics)."""
    ov, mr = [], []
    for (ec, ev), (qc, qv) in zip(ex_rows, q_rows):
        qc, qv = np.asarray(qc), np.asarray(qv)
        keep = np.abs(qv) > 0
        qc2, qv2 = qc[keep], qv[keep]
        kk = min(k, len(ec))
        if kk == 0:
            continue
        etop = ec[np.argsort(-np.abs(ev))[:kk]]
        qtop = (qc2[np.argsort(-np.abs(qv2))[:kk]] if len(qc2)
                else np.array([], dtype=int))
        inter = np.intersect1d(etop, qtop)
        ov.append(len(inter) / kk)
        emap = dict(zip(ec.tolist(), np.abs(ev).tolist()))
        tot = sum(emap[x] for x in etop.tolist())
        got = sum(emap.get(x, 0.0) for x in inter.tolist())
        mr.append(got / tot if tot > 0 else 0.0)
    return float(np.mean(ov)), float(np.mean(mr))


def digest(inst):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(inst['S']).tobytes())
    h.update(repr(inst['nz0']).encode())
    return h.hexdigest()[:16]


def local_hop(inst, in_rows):
    """The exact FP64 product of an engine's OWN hop input with S, so a
    hop can be scored without the drift (or the tie-break) carried in
    from earlier hops."""
    S = inst['S'].astype(np.float64)
    h = Hop()
    h.Y = np.asarray(rows_to_csr(in_rows, inst['N']) @ S)
    return h


def dump(path, outs, **meta):
    """Per-hop output rows of a run, for scoring offline."""
    z = dict(hops=np.int32(len(outs)))
    for h, rows in enumerate(outs):
        ptr = np.zeros(len(rows) + 1, np.int64)
        for i, (c, _) in enumerate(rows):
            ptr[i + 1] = ptr[i] + len(c)
        z[f'h{h}_ptr'] = ptr
        z[f'h{h}_col'] = (np.concatenate([np.asarray(c, np.int64) for c, _ in rows])
                          if ptr[-1] else np.zeros(0, np.int64))
        z[f'h{h}_val'] = (np.concatenate([np.asarray(v, np.float64) for _, v in rows])
                          if ptr[-1] else np.zeros(0))
    for k, v in meta.items():
        z[k] = v
    np.savez_compressed(path, **z)


def load(path):
    z = np.load(path, allow_pickle=False)
    outs = []
    for h in range(int(z['hops'])):
        p, c, v = z[f'h{h}_ptr'], z[f'h{h}_col'], z[f'h{h}_val']
        outs.append([(c[p[i]:p[i + 1]], v[p[i]:p[i + 1]]) for i in range(len(p) - 1)])
    return outs, z
