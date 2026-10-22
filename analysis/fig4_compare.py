#!/usr/bin/env python3
"""Fig. 4: ToME against the edge CPU and three FPGA accelerators on six
SuiteSparse graphs, all on the ZCU104: (a) time of three hops, (b) board
power.

k=100, M=256, three hops, every FPGA design clocked at 150 MHz.  Each hop
feeds every engine the exact FP64 frontier of the previous hop, so all
engines do identical work.  Bars are the time of the three hops: ToME and
the SpMV engines kernel only (the host's gather and pack of S for every
SpMV launch, results/zcu104/time/gbench2_*.json, is quoted in the text, not
drawn), the CPU its whole run, and Li et al.'s pipeline kernel + cache sync
+ A53 Top-k, each hop the minimum over all runs.

(b) is the 12-V board input (INA226): each design idling (hollow) and
running its timed work back to back (filled), mean over the six graphs,
the A53 with the PL clock stopped.

    python fig4_compare.py [out.pdf]
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch
import numpy as np

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results', 'zcu104')
OUT = sys.argv[1] if len(sys.argv) > 1 else 'fig4_compare.pdf'

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
    'legend.fontsize': 6.6, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2, 'ytick.major.size': 2, 'ytick.minor.size': 1.2,
    'axes.grid': True, 'axes.grid.axis': 'y', 'axes.axisbelow': True,
    'grid.color': '#e4e3df', 'grid.linewidth': 0.4,
    'hatch.linewidth': 0.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01,
})

INK, MUTED, TOME = '#0b0b0b', '#52514e', '#2a78d6'
GRAPHS = ['email', 'GD06_Java', 'ODLIS', 'SciMet', 'EPA', 'Erdos972']
SHORT = ['email\n1,133', 'GD06\n1,538', 'ODLIS\n2,909', 'SciMet\n3,084',
         'EPA\n4,772', 'Erdos\n5,488']
# entity -> (label, face, hatch); ToME keeps Fig. 3's blue, baselines are
# grey and told apart by hatch so the figure survives greyscale printing
ENG = {
    'a53':    ('A53 $\\times$4, sdt [10]', '#d9d8d4', ''),
    'tcad':   ('Li [8] + A53 Top-$k$', '#a9a8a3', '////'),
    'dac21':  ('Parravicini [18]', '#ffffff', '\\\\\\\\'),
    'dytopk': ('DyTopK [20], FP4', '#ffffff', 'xxxx'),
    'tome':   ('ToME', TOME, ''),
}


def rdir(eng):
    return 'time'


def load(eng, g):
    """the run, with each hop's time the minimum over it and any rerun"""
    p = os.path.join(RES, rdir(eng), f'{eng}_{g}.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    q = os.path.join(RES, rdir(eng), f'{eng}_{g}_rerun.json')
    if os.path.exists(q):
        e = json.load(open(q))
        for h, x in zip(d['hops'], e['hops']):
            for f in ('ms', 'kernel_ms', 'total_ms'):
                if f in h:
                    h[f] = min(h[f], x[f])
        if 'total_ms' in d:
            d['total_ms'] = sum(h['total_ms'] for h in d['hops'])
    return d


def total(eng, g):
    """(time of the three hops, worst-hop overlap)"""
    d = load(eng, g)
    if d is None:
        return None
    ov = min(h['overlap_tie'] for h in d['hops'])
    if eng == 'a53':
        return sum(h['ms'] for h in d['hops']), ov
    if eng == 'tcad':
        return d['total_ms'], ov
    return d['kernel_ms'], ov


fig, (ax, px) = plt.subplots(1, 2, figsize=(3.45, 1.5),
                             gridspec_kw=dict(width_ratios=[2.3, 1.0]))
x = np.arange(len(GRAPHS))
engines = list(ENG)
w = 0.84 / len(engines)
rows = {}
for j, e in enumerate(engines):
    vals = [total(e, g) for g in GRAPHS]
    rows[e] = vals
    lab, face, hatch = ENG[e]
    ax.bar(x + (j - (len(engines) - 1) / 2) * w, [v[0] if v else np.nan for v in vals],
           w * 0.92, color=face, hatch=hatch, edgecolor=TOME if e == 'tome' else INK,
           linewidth=0.4, zorder=2)
ax.set_yscale('log')
lo = min(v[0] for vals in rows.values() for v in vals if v)
ax.set_ylim(10 ** np.floor(np.log10(lo * 0.8)), None)
ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=12))
ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks(x)
ax.set_xticklabels(SHORT, linespacing=0.95, fontsize=6.3)
ax.tick_params(axis='x', pad=1)
ax.set_xlim(-0.55, len(GRAPHS) - 0.45)
ax.set_ylabel('time (ms)', labelpad=1)
ax.set_title('(a) time', pad=2, loc='left', x=0.02)
ax.tick_params(colors=MUTED, labelcolor=INK)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color(MUTED)

# (b) board power: idle (hollow) to running (filled), mean over the graphs
SHORTE = {'a53': 'A53', 'tcad': 'Li', 'dac21': 'Parravicini', 'dytopk': 'DyTopK',
          'tome': 'ToME'}


def power(eng):
    out = []
    for g in GRAPHS:
        d = json.load(open(os.path.join(RES, 'power', f'{eng}_{g}.json')))
        d = d.get('kernel', d)
        out.append((d['idle_w'], d['run_w']))
    return [sum(x) / len(x) for x in zip(*out)], [r for _, r in out]


pw = {e: power(e) for e in engines}
for y, e in enumerate(engines):
    (idle, run), runs = pw[e]
    c = TOME if e == 'tome' else INK
    px.plot([idle, run], [y, y], color=c, lw=1.0, zorder=2, solid_capstyle='butt')
    px.plot([min(runs), max(runs)], [y, y], color=c, lw=2.6, alpha=0.25, zorder=1,
            solid_capstyle='butt')
    px.scatter([idle], [y], s=14, facecolor='white', edgecolor=c, linewidth=0.8, zorder=3)
    px.scatter([run], [y], s=14, facecolor=c, edgecolor=c, linewidth=0.8, zorder=3)
px.set_yticks(range(len(engines)))
px.set_yticklabels([SHORTE[e] for e in engines])
px.set_ylim(len(engines) - 0.5, -0.5)
px.set_xlabel('board power (W)', labelpad=1)
px.set_title('(b) power', pad=2, loc='left', x=0.02)
px.grid(axis='x', color='#e4e3df', linewidth=0.4)
px.grid(axis='y', visible=False)
lo = min(min(v[0]) for v in pw.values())
hi = max(max(v[1]) for v in pw.values())
px.set_xlim(np.floor(lo * 2) / 2 - 0.15, np.ceil(hi * 2) / 2 + 0.15)
px.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
px.tick_params(colors=MUTED, labelcolor=INK)
px.tick_params(axis='y', length=0, pad=2)
for s in ('top', 'right', 'left'):
    px.spines[s].set_visible(False)
px.spines['bottom'].set_color(MUTED)

handles = [Patch(facecolor=face, hatch=hatch, edgecolor=TOME if e == 'tome' else INK,
                 lw=0.4, label=lab) for e, (lab, face, hatch) in ENG.items()]
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=0.8,
           handletextpad=0.35, borderaxespad=0.0)
fig.tight_layout(pad=0.1, w_pad=0.6, rect=(0, 0, 1, 0.78))
fig.savefig(OUT)
print('wrote', os.path.normpath(OUT))

# the numbers the text quotes
t = rows['tome']
for e, vals in rows.items():
    if e == 'tome':
        continue
    sp = [v[0] / tt[0] for v, tt in zip(vals, t) if v and tt]
    ov = [v[1] for v in vals if v]
    print(f'ToME vs {e:7s}: {min(sp):6.2f}-{max(sp):6.2f}x  worst-hop overlap '
          f'{min(ov):.3f}-{max(ov):.3f}')
print('ToME ms: ' + ', '.join(f'{g}={v[0]:.1f}' for g, v in zip(GRAPHS, t) if v))
for e in engines:
    (idle, run), runs = pw[e]
    print(f'{e:7s} board power: idle {idle:.2f} W  run {run:.2f} W  '
          f'(run {min(runs):.2f}-{max(runs):.2f} W over the graphs)')
