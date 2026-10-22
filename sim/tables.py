#!/usr/bin/env python3
"""Summarise a fidelity sweep (sweep.py) into the paper's numbers.

Prints Table II of the paper (codec ablation at five hops, mean of ten
graphs) and then the full breakdown: every codec at every (hops, k), ToME's
per-graph overlap and mass recall, each codec's traffic and fidelity
relative to single-tier FP8, and ToME's packet population.

    python tables.py [sweep_results.json]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(ROOT, 'results', 'fidelity', 'sweep_results.json')

R = json.load(open(DATA))
GRAPHS = ['email', 'GD06_Java', 'ODLIS', 'SciMet', 'EPA',
          'Erdos972', 'California', 'FA', 'EAT_RS', 'HEP-th-new']
CFGS = ['fp8', 'two', 'mixed', 'mixed_sexp', 'mixed_dyn']
LABEL = {'fp8': 'FP8 only', 'two': 'MT FP6/FP8', 'mixed': 'MT 3-tier',
         'mixed_sexp': '+ shared exp.', 'mixed_dyn': '+ dyn. tiers (ToME)'}

idx = {(r['graph'], r['hops'], r['k'], r['cfg']): r for r in R}


def mean(hops, k, cfg, key):
    # + 1e-12 rounds exact decimal ties up, as the paper does (MT at k=100
    # averages exactly 0.5485, printed as .549)
    return float(np.mean([idx[(g, hops, k, cfg)][key] for g in GRAPHS])) + 1e-12


# ------------------------------------------------------------- Table II
print('TABLE II  codec ablation at five hops, mean of ten graphs')
print('%-20s %18s   %18s   %7s' % ('', 'bits / non-zero', 'Top-k overlap', 'zero%'))
print('%-20s %6s%6s%6s   %6s%6s%6s   %7s' % ('codec   k =', 50, 100, 200, 50, 100, 200, 'k=200'))
for cfg in ('fp8', 'mixed', 'mixed_sexp', 'mixed_dyn'):
    print('%-20s %6.1f%6.1f%6.1f   %6.3f%6.3f%6.3f   %7.1f'
          % ((LABEL[cfg],) + tuple(mean(5, k, cfg, 'bits_per_nz') for k in (50, 100, 200))
             + tuple(mean(5, k, cfg, 'overlap') for k in (50, 100, 200))
             + (mean(5, 200, cfg, 'zero_pct'),)))
fp8, tome = mean(5, 200, 'fp8', 'bits_per_nz'), mean(5, 200, 'mixed_dyn', 'bits_per_nz')
print('ToME moves %.1f%% fewer bits than FP8 at k=200, five hops' % (100 * (fp8 - tome) / fp8))
dyn = [r for r in R if r['cfg'] == 'mixed_dyn' and r['graph'] in GRAPHS]
print('over all %d (graph, hops, k): ToME overlap >= %.3f, mass recall >= %.3f'
      % (len(dyn), min(r['overlap'] for r in dyn), min(r['mass_recall'] for r in dyn)))
print()

# ------------------------------------------------------ every (hops, k)
print('codec ablation at every (hops, k), mean over ten graphs')
print('%-20s %4s %4s %9s %7s %7s %8s %9s %9s' %
      ('codec', 'hop', 'k', 'bits/nz', 'occup', 'part%', 'zero%',
       'overlap', 'massrec'))
for hops in (2, 3, 5):
    for k in (50, 100, 200):
        for cfg in CFGS:
            print('%-20s %4d %4d %9.2f %7.2f %7.1f %8.2f %9.4f %9.4f' %
                  (LABEL[cfg], hops, k, mean(hops, k, cfg, 'bits_per_nz'),
                   mean(hops, k, cfg, 'occupancy'), mean(hops, k, cfg, 'partial_pct'),
                   mean(hops, k, cfg, 'zero_pct'), mean(hops, k, cfg, 'overlap'),
                   mean(hops, k, cfg, 'mass_recall')))
        print()

# ------------------------------------------------------------ per graph
for key, name in (('overlap', 'overlap'), ('mass_recall', 'mass recall')):
    print('ToME %s (%%) per graph' % name)
    print('%-5s %-4s ' % ('hop', 'k') + ' '.join('%10s' % g[:10] for g in GRAPHS)
          + ' %8s' % 'mean')
    for hops in (2, 3, 5):
        for k in (50, 100, 200):
            vals = [100 * idx[(g, hops, k, 'mixed_dyn')][key] for g in GRAPHS]
            print('%-5d %-4d ' % (hops, k) + ' '.join('%10.1f' % v for v in vals)
                  + ' %8.1f' % np.mean(vals))
    print()

# --------------------------------------------- relative to single-tier FP8
print('traffic and fidelity relative to single-tier FP8')
print('%-20s %4s %4s %12s %12s %12s' %
      ('codec', 'hop', 'k', 'traffic', 'd overlap', 'd massrec'))
for hops in (2, 5):
    for k in (50, 200):
        b_bits = mean(hops, k, 'fp8', 'bits_per_nz')
        b_ov, b_mr = mean(hops, k, 'fp8', 'overlap'), mean(hops, k, 'fp8', 'mass_recall')
        for cfg in CFGS:
            print('%-20s %4d %4d %+11.1f%% %+12.4f %+12.4f' %
                  (LABEL[cfg], hops, k,
                   100 * (mean(hops, k, cfg, 'bits_per_nz') - b_bits) / b_bits,
                   mean(hops, k, cfg, 'overlap') - b_ov,
                   mean(hops, k, cfg, 'mass_recall') - b_mr))
        print()

# ------------------------------------------------------ packet population
print('ToME packet population (mean over graphs)')
print('%4s %4s %9s %7s %7s %8s %8s %8s %8s' %
      ('hop', 'k', 'bits/nz', 'occup', 'part%', 'FP4%', 'FP6%', 'FP8%', 'zero%'))
for hops in (2, 3, 5):
    for k in (50, 100, 200):
        rs = [idx[(g, hops, k, 'mixed_dyn')] for g in GRAPHS]
        t4 = sum(x['tiers']['FP4'] for x in rs)
        t6 = sum(x['tiers']['FP6'] for x in rs)
        t8 = sum(x['tiers']['FP8'] for x in rs)
        tt = max(t4 + t6 + t8, 1)
        print('%4d %4d %9.2f %7.2f %7.1f %8.1f %8.1f %8.1f %8.2f' %
              (hops, k, mean(hops, k, 'mixed_dyn', 'bits_per_nz'),
               mean(hops, k, 'mixed_dyn', 'occupancy'),
               mean(hops, k, 'mixed_dyn', 'partial_pct'),
               100.0 * t4 / tt, 100.0 * t6 / tt, 100.0 * t8 / tt,
               mean(hops, k, 'mixed_dyn', 'zero_pct')))
