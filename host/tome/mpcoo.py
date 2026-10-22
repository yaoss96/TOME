"""Host-side MP-COO packer/decoder.

Mirrors the synthesised packer (greedy_pack_to_buf in hls/tome) bit for bit:
the host packs only the initial frontier, and decodes the kernel's output.
Layout, all packets 128 bits:

    127:126  tier        (FP8=0, FP6=1, FP4=2)
    125:118  row         panel-LOCAL row id (8 bits; was a 16-bit global id)
    117:110  sexp        shared exponent, biased by 128
    109:94   base_col    smallest column in the packet
    FP8      93 fmt, 92:90 cnt, then 5 slots of (8b value, 10b offset) from 89
    FP6      93 fmt, 92:90 cnt, then 6 slots of (6b value,  9b offset) from 89
    FP4      93:91 cnt,          then 7 slots of (4b value,  9b offset) from 90

Tier selection keys on the packet's dynamic range, compared in the exponent
domain so no divider is needed:  d = floor(log2 mmax) - floor(log2 mmin).
The shared exponent places mmax on the sub-format's ceiling.
"""
import math

PREC_FP8, PREC_FP6, PREC_FP4 = 0, 1, 2
NZ = {PREC_FP8: 5, PREC_FP6: 6, PREC_FP4: 7}
OFFS = {PREC_FP8: 1023, PREC_FP6: 511, PREC_FP4: 511}
SEXP_BIAS = 128

DEXP_FP4, DEXP_E2M3, DEXP_FP6, DEXP_E4M3 = 2, 5, 8, 17


def fexp_of(v):
    av = abs(v)
    return 0 if av == 0.0 else int(math.floor(math.log2(av)))


def span_exp(mmax, mmin):
    if mmin <= 0.0:
        return 255
    return fexp_of(mmax) - fexp_of(mmin)


def tier_for_range(mmax, mmin):
    d = span_exp(mmax, mmin)
    if d <= DEXP_FP4:
        return PREC_FP4
    if d <= DEXP_FP6:
        return PREC_FP6
    return PREC_FP8


def subfmt_for_range(prec, mmax, mmin):
    d = span_exp(mmax, mmin)
    if prec == PREC_FP8:
        return 1 if d <= DEXP_E4M3 else 0
    if prec == PREC_FP6:
        return 1 if d <= DEXP_E2M3 else 0
    return 0


def fmax_exp(prec, fmt):
    if prec == PREC_FP8:
        return -1 if fmt else 0
    if prec == PREC_FP6:
        return -2 if fmt else -1
    return -7


def fmt_params(prec, fmt):
    if prec == PREC_FP8:
        return (4, 3, 16) if fmt else (5, 2, 31)
    if prec == PREC_FP6:
        return (2, 3, 5) if fmt else (3, 2, 8)
    return (1, 2, 8)


def fscale2(v, n):
    if v == 0.0:
        return 0.0
    return math.ldexp(v, n)


def mfp_encode(val, eb, mb, bias):
    """bit-identical to the HLS mfp_encode"""
    if val == 0.0:
        return 0
    sgn = 1 if val < 0.0 else 0
    av = abs(val)
    exp_max = (1 << eb) - 1
    mant_max = (1 << mb) - 1
    fr, e = math.frexp(av)
    e -= 1
    fr *= 2.0
    ebv = e + bias
    if ebv > exp_max:
        return (sgn << (eb + mb)) | (exp_max << mb) | mant_max
    if ebv <= 0:
        mf = av / (2.0 ** (1.0 - bias)) * (1 << mb)
        m = min(int(mf + 0.5), mant_max)
        return (sgn << (eb + mb)) | m
    m = int((fr - 1.0) * (1 << mb) + 0.5)
    if m > mant_max:
        m = 0
        ebv += 1
        if ebv > exp_max:
            ebv, m = exp_max, mant_max
    return (sgn << (eb + mb)) | (ebv << mb) | m


def mfp_decode(enc, eb, mb, bias):
    mant_max = (1 << mb) - 1
    s = (enc >> (eb + mb)) & 1
    exp = (enc >> mb) & ((1 << eb) - 1)
    m = enc & mant_max
    if exp == 0 and m == 0:
        return 0.0
    if exp == 0:
        v = (m / float(1 << mb)) * (2.0 ** (1.0 - bias))
    else:
        v = (1.0 + m / float(1 << mb)) * (2.0 ** (exp - bias))
    return -v if s else v


def _sb(p, hi, lo, val):
    mask = (1 << (hi - lo + 1)) - 1
    return (p & ~(mask << lo)) | ((val & mask) << lo)


def _gb(p, hi, lo):
    return (p >> lo) & ((1 << (hi - lo + 1)) - 1)


def packetize(nz, row_base=0):
    """nz: list of (row, col, value), rows are GLOBAL; row_base is the panel
    base, subtracted to give the 8-bit panel-local id the kernel expects."""
    if not nz:
        return []
    el = sorted(nz, key=lambda x: (x[0], x[1]))
    pkts = []
    i, L = 0, len(el)
    while i < L:
        row = el[i][0]
        j = i + 1
        while j < L and el[j][0] == row:
            j += 1
        re_ = j
        ci = i
        while ci < re_:
            min_c = el[ci][1]
            mx = abs(el[ci][2])
            mn = mx
            cnt = 1
            cj = ci + 1
            while cj < re_ and cnt < NZ[PREC_FP4]:
                nm = abs(el[cj][2])
                cmax = max(nm, mx)
                cmin = nm if (0.0 < nm < mn) else mn
                p = tier_for_range(cmax, cmin)
                if cnt >= NZ[p]:
                    break
                if el[cj][1] - min_c > OFFS[p]:
                    break
                mx, mn = cmax, cmin
                cnt += 1
                cj += 1
            prec = tier_for_range(mx, mn)
            fmt = subfmt_for_range(prec, mx, mn)
            eb, mb, bias = fmt_params(prec, fmt)
            sh = fexp_of(mx) - fmax_exp(prec, fmt)
            sexp = sh + SEXP_BIAS

            p = 0
            p = _sb(p, 127, 126, prec)
            p = _sb(p, 125, 118, (row - row_base) & 0xFF)
            p = _sb(p, 117, 110, sexp & 0xFF)
            p = _sb(p, 109, 94, min_c & 0xFFFF)
            if prec == PREC_FP8:
                p = _sb(p, 93, 93, fmt)
                p = _sb(p, 92, 90, cnt)
                for e in range(cnt):
                    lc = el[ci + e][1] - min_c
                    ec = mfp_encode(fscale2(el[ci + e][2], -sh), eb, mb, bias)
                    bb = 89 - e * 18
                    p = _sb(p, bb, bb - 7, ec & 0xFF)
                    p = _sb(p, bb - 8, bb - 17, lc & 0x3FF)
            elif prec == PREC_FP6:
                p = _sb(p, 93, 93, fmt)
                p = _sb(p, 92, 90, cnt)
                for e in range(cnt):
                    lc = el[ci + e][1] - min_c
                    ec = mfp_encode(fscale2(el[ci + e][2], -sh), eb, mb, bias)
                    bb = 89 - e * 15
                    p = _sb(p, bb, bb - 5, ec & 0x3F)
                    p = _sb(p, bb - 6, bb - 14, lc & 0x1FF)
            else:
                p = _sb(p, 93, 91, cnt)
                for e in range(cnt):
                    lc = el[ci + e][1] - min_c
                    ec = mfp_encode(fscale2(el[ci + e][2], -sh), eb, mb, bias)
                    bb = 90 - e * 13
                    p = _sb(p, bb, bb - 3, ec & 0xF)
                    p = _sb(p, bb - 4, bb - 12, lc & 0x1FF)
            pkts.append(p)
            ci += cnt
        i = re_
    return pkts


def decode(pkt, row_base=0):
    """returns list of (row, col, value); rows are made global again"""
    prec = _gb(pkt, 127, 126)
    row = _gb(pkt, 125, 118) + row_base
    sexp = _gb(pkt, 117, 110)
    base = _gb(pkt, 109, 94)
    sh = sexp - SEXP_BIAS
    out = []
    if prec == PREC_FP8:
        fmt = _gb(pkt, 93, 93)
        cnt = _gb(pkt, 92, 90)
        eb, mb, bias = fmt_params(prec, fmt)
        for e in range(cnt):
            bb = 89 - e * 18
            enc = _gb(pkt, bb, bb - 7)
            off = _gb(pkt, bb - 8, bb - 17)
            out.append((row, base + off,
                        fscale2(mfp_decode(enc, eb, mb, bias), sh)))
    elif prec == PREC_FP6:
        fmt = _gb(pkt, 93, 93)
        cnt = _gb(pkt, 92, 90)
        eb, mb, bias = fmt_params(prec, fmt)
        for e in range(cnt):
            bb = 89 - e * 15
            enc = _gb(pkt, bb, bb - 5)
            off = _gb(pkt, bb - 6, bb - 14)
            out.append((row, base + off,
                        fscale2(mfp_decode(enc, eb, mb, bias), sh)))
    elif prec == PREC_FP4:
        cnt = _gb(pkt, 93, 91)
        eb, mb, bias = fmt_params(prec, 0)
        for e in range(cnt):
            bb = 90 - e * 13
            enc = _gb(pkt, bb, bb - 3)
            off = _gb(pkt, bb - 4, bb - 12)
            out.append((row, base + off,
                        fscale2(mfp_decode(enc, eb, mb, bias), sh)))
    return out


def roundtrip_check(nz, row_base=0):
    """max relative error and flush fraction over one packetise/decode cycle"""
    pk = packetize(nz, row_base)
    got = {}
    for p in pk:
        for (r, c, v) in decode(p, row_base):
            got[(r, c)] = v
    rel, flushed = [], 0
    for (r, c, v) in nz:
        d = got.get((r, c), 0.0)
        if v != 0.0 and d == 0.0:
            flushed += 1
        elif v != 0.0:
            rel.append(abs(d - v) / abs(v))
    rel.sort()
    return dict(packets=len(pk), n=len(nz),
                flush_pct=100.0 * flushed / max(len(nz), 1),
                med_rel=rel[len(rel) // 2] if rel else -1,
                max_rel=rel[-1] if rel else -1)
