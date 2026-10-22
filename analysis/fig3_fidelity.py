#!/usr/bin/env python3
"""Fig. 3: Top-k fidelity of the frontier codecs.

  (a) mean Top-k overlap vs. hops at k=200 for the four codecs of Table II
  (b) per-graph overlap at 5 hops, k=200: magnitude-tiered vs. ToME

Reads results/fidelity/sweep_results.json (sim/sweep.py: quantisation in
the loop, scored against exact FP64, full graph size) and writes a PDF with
TrueType (Type 42) fonts.

    python fig3_fidelity.py [out.pdf]
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'results', 'fidelity', 'sweep_results.json')
OUT = sys.argv[1] if len(sys.argv) > 1 else 'fig3_fidelity.pdf'

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2, 'ytick.major.size': 2,
    'axes.grid': True, 'axes.axisbelow': True,
    'grid.color': '#e4e3df', 'grid.linewidth': 0.4,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01,
})

INK, MUTED = '#0b0b0b', '#52514e'
GRAPHS = ['email', 'GD06_Java', 'ODLIS', 'SciMet', 'EPA',
          'Erdos972', 'California', 'FA', 'EAT_RS', 'HEP-th-new']
SHORT = ['email', 'GD06', 'ODLIS', 'SciMet', 'EPA',
         'Erdos', 'Calif.', 'FA', 'EAT', 'HEP']
# (cfg, label, colour, marker): ToME is categorical slot 1, the failing
# magnitude-tiered codec slot 2, its shared-exponent fix slot 3, and the FP8
# reference recedes to neutral ink.
CODECS = [('fp8', 'FP8 only', MUTED, 's'),
          ('mixed', 'MT 3-tier', '#eb6834', 'v'),
          ('mixed_sexp', '+ shared exp.', '#1baf7a', '^'),
          ('mixed_dyn', 'ToME', '#2a78d6', 'o')]

R = json.load(open(DATA))
idx = {(r['graph'], r['hops'], r['k'], r['cfg']): r for r in R}


def overlap(g, hops, k, cfg):
    return idx[(g, hops, k, cfg)]['overlap']


fig, (ax, bx) = plt.subplots(
    1, 2, figsize=(3.45, 1.75), gridspec_kw={'width_ratios': [1, 1.55]})

# ---- (a) degradation with depth --------------------------------------------
hops = (2, 3, 5)
for cfg, lab, col, mk in CODECS:
    ys = [np.mean([overlap(g, h, 200, cfg) for g in GRAPHS]) for h in hops]
    ax.plot(hops, ys, color=col, marker=mk, ms=3.2, lw=1.0, label=lab,
            markeredgecolor='white', markeredgewidth=0.4)
ax.set_xticks(hops)
ax.set_xlabel('hops')
ax.set_ylabel('Top-$k$ overlap')
ax.set_ylim(0, 1.04)
ax.set_title('(a) mean of 10 graphs, $k$=200', pad=2)

# ---- (b) every graph, deepest setting --------------------------------------
x = np.arange(len(GRAPHS))
w = 0.38
for i, (cfg, lab, col) in enumerate([('mixed', 'MT 3-tier', '#eb6834'),
                                     ('mixed_dyn', 'ToME', '#2a78d6')]):
    ys = [overlap(g, 5, 200, cfg) for g in GRAPHS]
    bx.bar(x + (i - 0.5) * w, ys, w * 0.9, color=col, label=lab, lw=0)
bx.set_xticks(x)
bx.set_xticklabels(SHORT, rotation=55, ha='right', rotation_mode='anchor')
bx.tick_params(axis='x', pad=1)
bx.set_ylim(0, 1.04)
bx.set_title('(b) per graph, 5 hops, $k$=200', pad=2)

# One legend for both panels: (b) reuses (a)'s colours for MT and ToME.
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=1.5, columnspacing=0.9,
           handletextpad=0.4, borderaxespad=0.0)

for a in (ax, bx):
    a.tick_params(colors=MUTED, labelcolor=INK)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        a.spines[s].set_color(MUTED)

fig.tight_layout(pad=0.1, w_pad=0.6, rect=(0, 0, 1, 0.9))
fig.savefig(OUT)
print('wrote', os.path.normpath(OUT))
