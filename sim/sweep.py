#!/usr/bin/env python3
"""Fidelity sweep of the frontier codecs (Table II, Fig. 3).

For every graph, hop count (2, 3, 5), k (50, 100, 200) and codec, the
frontier is propagated with quantisation inside the loop -- the decoded
output of hop t is the input of hop t+1 -- and scored against an exact FP64
run of the same chain by Top-k overlap and retained-mass recall.

S is the row-wise softmax over each node's graph neighbours (logit 1) and
self-loop (logit 3), built sparse, so every graph runs at full size.  Each
of the M query rows starts with 1-5 active states (seed 20261007).

Codecs (codec.py):
  fp32_exact  no quantisation (sanity check: overlap 1.0)
  fp8         single-tier FP8
  two         magnitude-tiered FP6/FP8
  mixed       magnitude-tiered FP4/FP6/FP8 (MT)
  mixed_sexp  MT + per-packet shared exponent
  mixed_dyn   ToME: shared exponent + dynamic-range tiers

    python sweep.py                   # all ten graphs, under a minute
    python sweep.py email Erdos972    # a subset

results/fidelity/sweep_results.json is the output of the first command.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codec import quantise_rows

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
GRAPHS = ['email', 'GD06_Java', 'ODLIS', 'SciMet', 'EPA',
          'Erdos972', 'California', 'FA', 'EAT_RS', 'HEP-th-new']
CFGS = [
    ('fp32_exact',  None,   None),
    ('fp8',         'abs',  ('FP8',)),
    ('two',         'abs',  ('FP6', 'FP8')),
    ('mixed',       'abs',  ('FP4', 'FP6', 'FP8')),
    ('mixed_sexp',  'sexp', ('FP4', 'FP6', 'FP8')),
    ('mixed_dyn',   'dyn',  ('FP4', 'FP6', 'FP8')),
]


def load_mtx(p):
    rows, cols = [], []
    with open(p) as f:
        hdr = True
        for ln in f:
            if ln.startswith('%'):
                continue
            a = ln.split()
            if hdr:
                N = int(a[0])
                hdr = False
                continue
            rows.append(int(a[0]) - 1)
            cols.append(int(a[1]) - 1)
    return N, np.array(rows), np.array(cols)


def build_S(N, r, c, selflogit=3.0, nbr=1.0):
    rr = np.concatenate([r, c, np.arange(N)])
    cc = np.concatenate([c, r, np.arange(N)])
    lg = np.concatenate([np.full(len(r), nbr), np.full(len(c), nbr),
                         np.full(N, selflogit)])
    A = coo_matrix((np.exp(lg), (rr, cc)), shape=(N, N)).tocsr()
    A.sum_duplicates()
    rs = np.asarray(A.sum(1)).ravel()
    rs[rs == 0] = 1.0
    return csr_matrix(A.multiply(1.0 / rs[:, None]))


def topk_rows(Pn, k):
    out = []
    for i in range(Pn.shape[0]):
        v = Pn[i]
        nz = np.nonzero(v)[0]
        if len(nz) > k:
            nz = nz[np.argsort(-np.abs(v[nz]))[:k]]
        o = np.sort(nz)
        out.append((o, v[o]))
    return out


def densify(rows, M, N):
    P = np.zeros((M, N))
    for i, (c, v) in enumerate(rows):
        if len(c):
            P[i, c] = v
    return P


def score(ex_rows, q_rows, k):
    """Top-k overlap and retained-mass recall against the exact rows"""
    ov, mr = [], []
    for (ec, ev), (qc, qv) in zip(ex_rows, q_rows):
        keep = np.abs(qv) > 0
        qc2, qv2 = qc[keep], qv[keep]
        kk = min(k, len(ec))
        if kk == 0:
            continue
        etop = ec[np.argsort(-np.abs(ev))[:kk]]
        qtop = qc2[np.argsort(-np.abs(qv2))[:kk]] if len(qc2) else np.array([], dtype=int)
        inter = np.intersect1d(etop, qtop)
        ov.append(len(inter) / kk)
        emap = dict(zip(ec, np.abs(ev)))
        tot = sum(emap[x] for x in etop)
        got = sum(emap.get(x, 0.0) for x in inter)
        mr.append(got / tot if tot > 0 else 0.0)
    return float(np.mean(ov)), float(np.mean(mr))


def run_one(N, S, M, k, hops, mode, tiers, seed=20261007):
    rng = np.random.default_rng(seed)
    P0 = np.zeros((M, N))
    for i in range(M):
        act = rng.choice(N, size=int(rng.integers(1, 6)), replace=False)
        w = rng.random(len(act))
        P0[i, act] = w / w.sum()

    Pex, Pq = P0.copy(), P0.copy()
    last = None
    for h in range(hops):
        ex_rows = topk_rows(Pex @ S, k)
        Pex = densify(ex_rows, M, N)
        q_in = topk_rows(Pq @ S, k)
        if mode is None:
            dq_rows, st = q_in, None
        else:
            dq_rows, st = quantise_rows(q_in, mode, tiers)
        Pq = densify(dq_rows, M, N)
        ov, mr = score(ex_rows, dq_rows, k)
        last = (ov, mr, st)
    return last


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('graphs', nargs='*', default=GRAPHS)
    ap.add_argument('--mtx-dir', default=os.path.join(ROOT, 'data', 'graphs'),
                    help='directory of <graph>.mtx (data/get_graphs.py)')
    ap.add_argument('--m', type=int, default=16, help='query rows M')
    ap.add_argument('--out', default='sweep_results.json')
    a = ap.parse_args()

    recs = []
    for g in a.graphs:
        path = os.path.join(a.mtx_dir, g + '.mtx')
        if not os.path.exists(path):
            print('skip (missing): ' + path)
            continue
        t0 = time.time()
        N, r, c = load_mtx(path)
        S = build_S(N, r, c)
        print('# %-13s N=%-6d nnz=%-8d (%.1fs)'
              % (g, N, S.nnz, time.time() - t0), flush=True)
        for hops in (2, 3, 5):
            for k in (50, 100, 200):
                for nm, mode, tiers in CFGS:
                    t1 = time.time()
                    ov, mr, st = run_one(N, S, a.m, k, hops, mode, tiers)
                    rec = dict(graph=g, N=int(N), nnz=int(S.nnz), M=a.m,
                               hops=hops, k=k, cfg=nm, overlap=ov,
                               mass_recall=mr, secs=round(time.time() - t1, 2))
                    if st:
                        rec.update(bits_per_nz=st['bits_per_nz'],
                                   occupancy=st['occupancy'],
                                   zero_pct=st['zero_pct'],
                                   partial_pct=st['partial_pct'],
                                   tiers=st['tiers'])
                    recs.append(rec)
                    print('  %-13s h=%d k=%-3d %-12s ov=%.4f mr=%.4f %s'
                          % (g, hops, k, nm, ov, mr,
                             ('%.2f b/nz z=%.1f%%' % (st['bits_per_nz'], st['zero_pct']))
                             if st else ''), flush=True)
        with open(a.out, 'w') as f:
            json.dump(recs, f, indent=1)
    print('wrote %s (%d records)' % (a.out, len(recs)))


if __name__ == '__main__':
    main()
