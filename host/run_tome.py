#!/usr/bin/env python3
"""Run ToME on the ZCU104 and score every hop against the exact FP64 chain.

Builds the instance of problem.py from a SuiteSparse graph, programs the
bitstream, sets and verifies the PL clock, and runs --hops hops of
P <- TopK(P S).  The host packs only the initial frontier; each later hop
consumes the kernel's own output packets, unless --force feeds every hop
the exact FP64 frontier of the previous hop (how Fig. 4 gives every
engine identical work).  Kernel time is ap_start to ap_done, the minimum
over --reps repetitions, which must produce identical output.

Fig. 4, e.g. on email:
    sudo ... python3 run_tome.py email.mtx --bit tome_zcu150.bit --mhz 150 \\
        --k 100 --hops 3 --force --out tome_email.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tome import driver as D
from tome import mpcoo as MP
from tome import problem as I
from pynq import Overlay, allocate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mtx', help='SuiteSparse graph, Matrix Market format')
    ap.add_argument('--bit', default='tome_zcu150.bit',
                    help='bitstream; its .hwh must sit next to it')
    ap.add_argument('--mhz', type=float, default=150.0, help='PL clock')
    ap.add_argument('--nmax', type=int, default=0, help='keep the first nodes only')
    ap.add_argument('--m', type=int, default=256, help='query rows M')
    ap.add_argument('--k', type=int, default=100)
    ap.add_argument('--tn', type=int, default=256, help='column block T_N')
    ap.add_argument('--rs', type=int, default=256, help='rows per panel')
    ap.add_argument('--hops', type=int, default=3)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--force', action='store_true',
                    help='feed every hop the exact FP64 frontier of the previous '
                         'hop, so every engine does identical work')
    ap.add_argument('--out', default=None, help='JSON of the run')
    ap.add_argument('--dump', default=None, help='npz of per-hop outputs')
    a = ap.parse_args()

    P = I.build(a.mtx, a.nmax, a.m)
    n, m, K = P['N'], P['M'], a.k
    sigma, Bp = D.encode_s(P['S'])
    nhw = Bp.shape[1]
    ex = I.exact_chain(P, K, a.hops)

    ol = Overlay(a.bit)
    fclk = D.set_mhz(a.mhz)
    drv = D.Kernel.from_overlay(ol)
    bb = allocate(shape=(n * nhw // 4,), dtype=np.uint32)
    bb[:] = Bp.reshape(-1).view(np.uint32)
    del Bp
    ab = allocate(shape=(D.MAX_A_PKT * 4,), dtype=np.uint32)
    rb = allocate(shape=(D.RES_BUF_SIZE * 4,), dtype=np.uint32)
    bbs = [bb] * len(D.REGS['bp'])
    print(f"# ToME {os.path.basename(a.bit)}  {os.path.basename(a.mtx)} N={n} M={m} "
          f"k={K} hops={a.hops} T_N={a.tn}  sigma={sigma}  fclk {fclk:.2f} MHz  "
          f"inst={I.digest(P)}", flush=True)

    cur = P['nz0']
    outs = []
    npan = (m + a.rs - 1) // a.rs
    hops = []
    try:
        for h in range(a.hops):
            if a.force and h > 0:
                cur = [(i, int(c), float(v)) for i, (cs, vs) in enumerate(ex[h - 1])
                       for c, v in zip(cs, vs)]
            panels = [[] for _ in range(npan)]
            for (r, c, v) in cur:
                panels[r // a.rs].append((r, c, v))
            packed = [MP.packetize(p, pi * a.rs) for pi, p in enumerate(panels)]
            best, nxt, digests, times = None, [], [], []
            for _ in range(a.reps):
                tot, nxt, nout = 0.0, [], 0
                for pi, rp in enumerate(packed):
                    if not rp:
                        continue
                    ab[:len(rp) * 4] = 0
                    D.pkts2buf(ab, rp)
                    rb[:] = D.SENTINEL_W
                    rb[0] = 1
                    tot += drv.call(ab, bbs, rb, len(rp), pi * a.rs, m, n, nhw,
                                    a.rs, a.tn, K, K, sigma)
                    raw, nr, _ = D.extract_pkts(rb)
                    nout += nr
                    for pk in raw:
                        nxt.extend(MP.decode(pk, pi * a.rs))
                best = tot if best is None else min(best, tot)
                times.append(tot * 1e3)
                digests.append(hashlib.sha256(repr(sorted(nxt)).encode()).hexdigest()[:16])
            rows = I.nz_to_rows(nxt, m)
            outs.append(rows)
            ov, mr = I.score(ex[h], rows, K)
            tov, tmr = I.score_tie(ex[h], rows, K)
            hops.append(dict(hop=h + 1, kernel_ms=best * 1e3, nz_out=len(nxt),
                             out_pkts=nout, overlap=ov, mass_recall=mr, overlap_tie=tov,
                             mass_tie=tmr, rep_ms=times, out_sha=digests[0],
                             reps_identical=len(set(digests)) == 1))
            print(f"  hop {h + 1}: kernel {best * 1e3:8.2f} ms  nnz {len(nxt)}  "
                  f"pkts {nout}  overlap {ov:.4f}  mass {mr:.4f}  tie-aware {tov:.4f}/{tmr:.4f}  "
                  f"out {digests[0]} reps {'identical' if len(set(digests)) == 1 else 'DIFFER'} "
                  f"spread {100 * (max(times) / min(times) - 1):.2f}%", flush=True)
            cur = nxt
            if not cur:
                break
    finally:
        ab.freebuffer(); rb.freebuffer(); bb.freebuffer()

    res = dict(engine='tome_zcu104', graph=os.path.basename(a.mtx), N=n,
               n_graph=P['n_graph'], M=m, k=K, tn=a.tn, hops_n=a.hops,
               fclk_mhz=fclk, bit=os.path.basename(a.bit), inst=I.digest(P),
               kernel_ms=sum(x['kernel_ms'] for x in hops), hops=hops)
    print(f"  total kernel {res['kernel_ms']:.2f} ms", flush=True)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(res, f, indent=1)
        print('wrote ' + a.out)
    if a.dump:
        I.dump(a.dump, outs, inst=I.digest(P), k=np.int32(K))
        print('dumped ' + a.dump)


if __name__ == '__main__':
    main()
