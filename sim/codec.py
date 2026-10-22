"""Software model of the frontier codecs compared in the paper.

pack_row() packs one row's column-sorted non-zeros into 128-bit packets and
returns what the decoder recovers, so that quantisation can sit inside the
propagation loop (sweep.py).  Three packing modes:

  'abs'   magnitude-tiered (MT): each value's tier (FP4/FP6/FP8) is set by
          its absolute magnitude, no shared exponent
  'sexp'  MT tiers plus a per-packet shared exponent that places the
          packet's largest value at the sub-format's ceiling
  'dyn'   ToME / MP-COO: shared exponent, and the tier chosen by the
          packet's dynamic range (max/min magnitude)

The minifloat encode/decode (enc_dec) rounds exactly as the HLS codec.
Packet capacities and column-offset spans follow MP-COO: 5 FP8, 6 FP6 or
7 FP4 slots with 10/9/9-bit offsets.
"""
import math
import numpy as np

FMT = {'E5M2': (5, 2, 31), 'E4M3': (4, 3, 16),
       'E3M2': (3, 2, 8),   'E2M3': (2, 3, 5),
       'E1M2': (1, 2, 8)}

TIER_SUBS = {'FP8': ('E5M2', 'E4M3'), 'FP6': ('E3M2', 'E2M3'), 'FP4': ('E1M2',)}
NZ = {'FP8': 5, 'FP6': 6, 'FP4': 7}
OFFS = {'FP8': 1023, 'FP6': 511, 'FP4': 511}
ORD = {'FP4': 0, 'FP6': 1, 'FP8': 2}
TAU4, TAU6 = 0.013672, 0.875


def fmt_max(eb, mb, bias):
    return (2.0 - 2.0 ** -mb) * 2.0 ** ((1 << eb) - 1 - bias)


def fmt_min(eb, mb, bias):
    """smallest non-zero (subnormal) magnitude"""
    return (1.0 / (1 << mb)) * 2.0 ** (1 - bias)


FMAX = {k: fmt_max(*v) for k, v in FMT.items()}
FMIN = {k: fmt_min(*v) for k, v in FMT.items()}
# dynamic range each sub-format can span inside one packet
FDYN = {k: FMAX[k] / FMIN[k] for k in FMT}


def enc_dec(av, eb, mb, bias):
    if av == 0.0:
        return 0.0
    exp_max = (1 << eb) - 1
    mant_max = (1 << mb) - 1
    fr, e = math.frexp(av)
    e -= 1
    fr *= 2.0
    ebv = e + bias
    if ebv > exp_max:
        exp, m = exp_max, mant_max
    elif ebv <= 0:
        mf = av / (2.0 ** (1.0 - bias)) * (1 << mb)
        m = min(int(mf + 0.5), mant_max)
        exp = 0
    else:
        mf = (fr - 1.0) * (1 << mb)
        m = int(mf + 0.5)
        if m > mant_max:
            m = 0
            ebv += 1
            if ebv > exp_max:
                ebv, m = exp_max, mant_max
        exp = ebv
    if exp == 0 and m == 0:
        return 0.0
    if exp == 0:
        return (m / (1 << mb)) * 2.0 ** (1.0 - bias)
    return (1.0 + m / (1 << mb)) * 2.0 ** (exp - bias)


def tier_abs(v):
    """magnitude-tiered (MT) policy: tier from the absolute magnitude"""
    if v <= TAU4:
        return 'FP4'
    if v <= TAU6:
        return 'FP6'
    return 'FP8'


def pick_sub_abs(t, mmax):
    if t == 'FP8':
        return 'E5M2' if mmax > 0.93 else 'E4M3'
    if t == 'FP6':
        if mmax > 0.46 or mmax < 0.06:
            return 'E3M2'
        return 'E2M3'
    return 'E1M2'


def pick_sub_dyn(t, mmax, mmin):
    """with a shared exponent only the WITHIN-PACKET spread matters, so
    spend bits on mantissa whenever the spread allows it"""
    span = mmax / mmin if mmin > 0 else float('inf')
    subs = TIER_SUBS[t]
    best = None
    for s in subs:
        if FDYN[s] >= span:
            # prefer more mantissa among those that fit
            if best is None or FMT[s][1] > FMT[best][1]:
                best = s
    return best if best else max(subs, key=lambda s: FDYN[s])


def pack_row(cols, vals, mode, tiers_allowed=('FP4', 'FP6', 'FP8')):
    """mode: 'abs'   = magnitude-tiered (MT) scheme
             'sexp'  = + per-packet shared exponent at the format ceiling
             'dyn'   = sexp + tier chosen by within-packet dynamic range
    """
    n = len(cols)
    i = 0
    out = []
    lo = tiers_allowed[0]
    while i < n:
        if mode == 'dyn':
            # grow the packet, then pick the narrowest tier whose dynamic
            # range covers the group and whose slot count fits
            S = [i]
            cmin = cmax = cols[i]
            j = i + 1
            while j < n and len(S) < NZ['FP4']:
                if (max(cmax, cols[j]) - min(cmin, cols[j])) > OFFS['FP4']:
                    break
                S.append(j)
                cmin, cmax = min(cmin, cols[j]), max(cmax, cols[j])
                j += 1
            while True:
                mags = [abs(vals[x]) for x in S]
                mmax, mmin = max(mags), min([m for m in mags if m > 0] or [1.0])
                span = mmax / mmin
                t = None
                for cand in ('FP4', 'FP6', 'FP8'):
                    if cand not in tiers_allowed:
                        continue
                    sub = pick_sub_dyn(cand, mmax, mmin)
                    if FDYN[sub] >= span and len(S) <= NZ[cand] and \
                       (cmax - cmin) <= OFFS[cand]:
                        t = cand
                        break
                if t is not None:
                    break
                if len(S) == 1:
                    t = tiers_allowed[-1]
                    break
                S.pop()                     # shed the outlier and retry
                j -= 1
                cmin, cmax = min(cols[x] for x in S), max(cols[x] for x in S)
            sub = pick_sub_dyn(t, mmax, mmin)
        else:
            t = tier_abs(abs(vals[i]))
            if ORD[t] < ORD[lo]:
                t = lo
            S = [i]
            cmin = cmax = cols[i]
            mmax = abs(vals[i])
            j = i + 1
            while j < n:
                cand = abs(vals[j])
                ct = tier_abs(cand)
                if ORD[ct] < ORD[lo]:
                    ct = lo
                nt = t if ORD[ct] <= ORD[t] else ct
                ncmin, ncmax = min(cmin, cols[j]), max(cmax, cols[j])
                if len(S) + 1 > NZ[nt] or (ncmax - ncmin) > OFFS[nt]:
                    break
                t = nt
                S.append(j)
                cmin, cmax = ncmin, ncmax
                mmax = max(mmax, cand)
                j += 1
            sub = pick_sub_abs(t, mmax)

        eb, mb, bias = FMT[sub]
        if mode in ('sexp', 'dyn'):
            mags = [abs(vals[x]) for x in S]
            mmax = max(mags)
            # place the packet max at the format's ceiling
            sh = (math.floor(math.log2(mmax)) -
                  math.floor(math.log2(FMAX[sub]))) if mmax > 0 else 0
        else:
            sh = 0
        sc = 2.0 ** -sh
        rec = []
        for x in S:
            a = abs(vals[x]) * sc
            d = enc_dec(a, eb, mb, bias) / sc
            rec.append((cols[x], math.copysign(d, vals[x])))
        out.append(dict(tier=t, sub=sub, n=len(S), rec=rec,
                        full=(len(S) == NZ[t])))
        i = j if mode != 'dyn' else S[-1] + 1
    return out


def quantise_rows(rows, mode, tiers_allowed=('FP4', 'FP6', 'FP8')):
    """returns (dequantised rows, packet statistics)"""
    tot_nz = tot_pk = zeros = occ = partial = 0
    tiers = {'FP4': 0, 'FP6': 0, 'FP8': 0}
    out = []
    for cols, vals in rows:
        pk = pack_row(np.asarray(cols), np.asarray(vals, dtype=np.float64),
                      mode, tiers_allowed)
        c2, v2 = [], []
        for p in pk:
            tot_pk += 1
            occ += p['n']
            tiers[p['tier']] += 1
            if not p['full']:
                partial += 1
            for c, v in p['rec']:
                tot_nz += 1
                if v == 0.0:
                    zeros += 1
                c2.append(c)
                v2.append(v)
        out.append((np.asarray(c2), np.asarray(v2)))
    st = dict(bits_per_nz=tot_pk * 128 / max(tot_nz, 1),
              occupancy=occ / max(tot_pk, 1),
              packets=tot_pk, nz=tot_nz,
              zero_pct=100 * zeros / max(tot_nz, 1),
              partial_pct=100 * partial / max(tot_pk, 1),
              tiers=tiers)
    return out, st
