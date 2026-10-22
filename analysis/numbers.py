#!/usr/bin/env python3
"""Recompute the measured numbers quoted in Sections IV-C and IV-D.

Inputs, all under results/zcu104 (ZCU104 board):
  board_100mhz/  B0 and ToME at 100 MHz, email and Erdos972 at k=200, and
                 the A53 with fused (sparse_dot_topn) and unfused Top-k
  time/          Fig. 4: three hops at k=100, M=256, FPGA designs at 150 MHz;
                 gbench2_*: the A53's gather+pack of S for the SpMV engines
  power/         Fig. 4(b): 12-V board power (INA226) while each design
                 repeats its timed work; the SpMV engines' kernel and host
                 packing are measured separately

Energy of a run is its power times its time.  Times are the minimum per hop
over all runs of a design, as in Fig. 4.

    python numbers.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results', 'zcu104')
G = ['email', 'GD06_Java', 'ODLIS', 'SciMet', 'EPA', 'Erdos972']
E = ['a53', 'tcad', 'dac21', 'dytopk', 'tome']
NAME = dict(a53='A53 x4 (sparse_dot_topn)', tcad='Li et al. + A53 Top-k',
            dac21='Parravicini et al.', dytopk='DyTopK (FP4)', tome='ToME')


def jl(*p):
    return json.load(open(os.path.join(RES, *p)))


def rng(xs, fmt='{:.1f}'):
    a, b = fmt.format(min(xs)), fmt.format(max(xs))
    return a if a == b else f'{a}-{b}'


def run(eng, g):
    """(seconds of the three hops, worst-hop tie-aware overlap)"""
    runs = [jl('time', f'{eng}_{g}.json')]
    if os.path.exists(os.path.join(RES, 'time', f'{eng}_{g}_rerun.json')):
        runs.append(jl('time', f'{eng}_{g}_rerun.json'))
    f = 'ms' if eng == 'a53' else ('total_ms' if eng == 'tcad' else 'kernel_ms')
    t = sum(min(x) for x in zip(*[[h[f] for h in r['hops']] for r in runs])) / 1e3
    return t, min(h['overlap_tie'] for h in runs[0]['hops'])


def pack(eng, g):
    """seconds the A53 spends gathering and packing S for an SpMV engine"""
    return jl('time', f'gbench2_{eng}_{g}.json')['total_pack_ms'] / 1e3


def power(eng, g):
    """(idle W, running W, host-pack W or None)"""
    d = jl('power', f'{eng}_{g}.json')
    if eng in ('dac21', 'dytopk'):
        return d['kernel']['idle_w'], d['kernel']['run_w'], d['host']['run_w']
    return d['idle_w'], d['run_w'], None


# ------------------------------------------------------- IV-C, 100 MHz
s = jl('board_100mhz', 'summary.json')


def total(prefix):
    return [v['total_ms'] for k, v in s.items() if k.startswith(prefix)][0]


b0 = sum(r['kernel_ms'] for r in jl('board_100mhz', 'b0_email_k200.json'))
tome = total('tome_zcu104|tome_zcu.bit|email.mtx|N=1133|k=200')
print('IV-C, ZCU104 at 100 MHz')
print(f'  email, k=200, three hops: B0 {b0:.1f} ms, ToME {tome:.1f} ms -> {b0 / tome:.2f}x')
hop1 = s['tome_zcu104|tome_zcu.bit|email.mtx|N=1133|k=200']['hop_ms'][0]
print(f'  ToME first hop on email: {hop1:.2f} ms = {hop1 * 1e-3 * 100e6 / 1e6:.2f} M cycles')
seg = [total(f'tome_zcu104|tome_zcu_ftk.bit|{w}') / total(f'tome_zcu104|tome_zcu.bit|{w}')
       for w in ('email.mtx|N=1133|k=200', 'Erdos972.mtx|N=5488|k=200', 'email.mtx|N=1024|k=100')]
print(f'  segmented Top-k buffer vs full scan: {rng(seg, "{:.2f}")}x slower')
unf = total('a53_scipy||email.mtx|N=1133|k=200') / total('a53_sdt||email.mtx|N=1133|k=200')
print(f'  A53, separate product and selection vs fused sparse_dot_topn: {unf:.2f}x slower')

# ------------------------------------------------------ IV-D, 150 MHz
T = {e: [run(e, g) for g in G] for e in E}
tt = [t for t, _ in T['tome']]
print('\nIV-D / Fig. 4, ZCU104, FPGA designs at 150 MHz, k=100, three hops')
print('  ToME ms: ' + ', '.join(f'{g} {t * 1e3:.1f}' for g, t in zip(G, tt)))
for e in E[:-1]:
    sp = [t / x for (t, _), x in zip(T[e], tt)]
    print(f'  ToME is {rng(sp, "{:.2f}"):>11s}x faster than {NAME[e]}')
for e in E:
    print(f'  worst-hop overlap {NAME[e]:26s} {rng([o for _, o in T[e]], "{:.3f}")}')
hp = [pack(e, g) for e in ('dac21', 'dytopk') for g in G]
print(f'  A53 gather+pack for the SpMV engines: {rng(hp)} s per run')
a53 = [t for t, _ in T['a53']]
print('  ... that is %sx the A53\'s time for the whole product'
      % rng([pack(e, g) / a for e in ('dac21', 'dytopk') for g, a in zip(G, a53)]))
print('  with it counted, ToME is %sx faster than both SpMV engines'
      % rng([(pack(e, g) + run(e, g)[0]) / x for e in ('dac21', 'dytopk')
             for g, x in zip(G, tt)], '{:.0f}'))

# ------------------------------------------------------------ power
P = {e: [power(e, g) for g in G] for e in E}
mean = lambda xs: sum(xs) / len(xs)
print('\nFig. 4(b), 12-V board power (W), mean over the graphs')
for e in E:
    i, r = mean([x[0] for x in P[e]]), mean([x[1] for x in P[e]])
    print(f'  {NAME[e]:26s} idle {i:5.2f}  running {r:5.2f} ({rng([x[1] for x in P[e]], "{:.2f}")})'
          f'  activity +{rng([x[1] - x[0] for x in P[e]], "{:.2f}")}')
act = {e: [x[1] - x[0] for x in P[e]] for e in E}
least = all(act['tome'][j] < min(act[e][j] for e in E if e != 'tome') for j in range(len(G)))
print('  ToME\'s activity power is the least of all designs on every graph:', least)

Ek = {e: [p[1] * t for p, (t, _) in zip(P[e], T[e])] for e in E}
Eh = {e: [p[1] * t + (p[2] or 0) * (pack(e, g) if p[2] else 0)
          for p, (t, _), g in zip(P[e], T[e], G)] for e in E}
print('\nEnergy per three-hop run')
print('  ToME: ' + rng([x * 1e3 for x in Ek['tome']], '{:.0f}') + ' mJ')
for e in E[:-1]:
    rk = [x / y for x, y in zip(Ek[e], Ek['tome'])]
    line = f'  ToME uses {rng(rk):>9s}x less energy than {NAME[e]}'
    if e in ('dac21', 'dytopk'):
        line += ' (kernel), %sx with host packing' % rng([x / y for x, y in zip(Eh[e], Ek['tome'])])
    print(line)
