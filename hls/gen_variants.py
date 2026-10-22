#!/usr/bin/env python3
"""Generate ToME from the B0 baseline source (hls/b0).

Each flag is one design decision of the paper, applied as a source
transformation of B0 so that every step can be synthesized on its own:

  --fastexp  replace powf(2,n)/frexpf in the minifloat codec with
             exponent-field arithmetic.  Every power of two in the codec
             has an integer exponent, so it is a field add rather than a
             floating-point pow core, of which B0 instantiates two in each
             of the decoder's seven slots.
  --sexp     MP-COO: a per-packet shared exponent, paid for by narrowing
             the 16-bit global row ID to an 8-bit panel-local ID, plus tier
             selection by the packet's dynamic range.  The 34-bit base
             header (tier, row, base column) and every slot layout keep
             their size.
  --bq       store the dense operand S in FP8 E4M3 with one global shared
             exponent, so a 128-bit beat carries 16 operands and the PEs
             run 16 MAC lanes instead of 4.

ToME is --fastexp --sexp --bq; hls/tome was generated with

    python gen_variants.py --out tome --fastexp --sexp --bq --period 6.667

(--period is the HLS clock target in ns; 6.667 ns = 150 MHz).
"""
import argparse
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SEXP_HELPERS = r'''
/* ================================================================
 *  MP-COO helpers: exponent-field arithmetic
 *
 *  fscale2 / fexp_of / fmant_of replace powf(2,n) and frexpf.  Every
 *  power of two in the MFP codec has an INTEGER exponent, so it is a
 *  field add -- not a floating-point pow.
 *
 *  The per-packet shared exponent: P(t) is row-stochastic and
 *  CONTRACTS with every hop, so magnitudes drift into and below the
 *  subnormal range of any fixed low-bit float.  Re-centring each
 *  packet on its own exponent is what makes the FP6/FP4 tiers usable.
 *  It is paid for by narrowing the 16-bit global row ID to an 8-bit
 *  panel-local ID (row_base is already a kernel argument), so the
 *  34-bit base header and every slot layout stay bit-for-bit
 *  identical.
 * ================================================================ */
inline int fexp_of(float v) {
#pragma HLS INLINE
    union { float f; unsigned u; } cv; cv.f = v;
    int e = (int)((cv.u >> 23) & 0xFFu) - 127;
    return ((cv.u & 0x7FFFFFFFu) == 0u) ? 0 : e;
}

inline float fscale2(float v, int n) {
#pragma HLS INLINE
    union { float f; unsigned u; } cv; cv.f = v;
    if ((cv.u & 0x7FFFFFFFu) == 0u) return 0.0f;
    int e = (int)((cv.u >> 23) & 0xFFu) + n;
    if (e <= 0)   return 0.0f;
    if (e >= 255) e = 254;
    cv.u = (cv.u & 0x807FFFFFu) | ((unsigned)e << 23);
    return cv.f;
}

inline float fmant_of(float av, int &e_out) {
#pragma HLS INLINE
    union { float f; unsigned u; } cv; cv.f = av;
    e_out = (int)((cv.u >> 23) & 0xFFu) - 127;
    cv.u = (cv.u & 0x807FFFFFu) | (127u << 23);
    return cv.f;
}

#define SEXP_BIAS 128

/* ----------------------------------------------------------------
 *  floor(log2(max representable)) per sub-format.  The shared
 *  exponent must land the packet's largest value on the format's
 *  CEILING, not on [1,2): FP4/E1M2 tops out at 1.75*2^-7, so
 *  normalising to [1,2) would saturate every FP4 packet.
 * ---------------------------------------------------------------- */
inline int fmax_exp(int prec, int fmt) {
#pragma HLS INLINE
    if (prec == PREC_FP8) return fmt ? -1 : 0;   /* E4M3 : E5M2 */
    if (prec == PREC_FP6) return fmt ? -2 : -1;  /* E2M3 : E3M2 */
    return -7;                                   /* E1M2        */
}

/* ----------------------------------------------------------------
 *  With a shared exponent the absolute magnitude no longer matters;
 *  what a tier must cover is the packet's INTERNAL spread.  These
 *  are the dynamic ranges (max/min representable) of each
 *  sub-format: E1M2 7, E2M3 60, E3M2 448, E4M3 245760, E5M2 7.5e9.
 *
 *  The spread is compared in the EXPONENT domain: d = floor(log2 mmax)
 *  - floor(log2 mmin) bounds the ratio by 2^d <= span < 2^(d+1), so an
 *  integer subtract of two exponent fields replaces a floating-point
 *  divide -- which, unrolled over seven slots in every PE's packer,
 *  would otherwise instantiate a divider per slot.
 *  Thresholds: E1M2 spans 7 (d<=2), E2M3 60 (d<=5), E3M2 448 (d<=8),
 *  E4M3 245760 (d<=17).
 * ---------------------------------------------------------------- */
#define DEXP_FP4   2
#define DEXP_E2M3  5
#define DEXP_FP6   8
#define DEXP_E4M3  17

inline int span_exp(float mmax, float mmin) {
#pragma HLS INLINE
    if (mmin <= 0.0f) return 255;
    return fexp_of(mmax) - fexp_of(mmin);
}

inline int tier_for_range(float mmax, float mmin) {
#pragma HLS INLINE
    int d = span_exp(mmax, mmin);
    if (d <= DEXP_FP4) return PREC_FP4;
    if (d <= DEXP_FP6) return PREC_FP6;
    return PREC_FP8;
}

/* prefer the wider mantissa whenever it still covers the spread */
inline int subfmt_for_range(int prec, float mmax, float mmin) {
#pragma HLS INLINE
    int d = span_exp(mmax, mmin);
    if (prec == PREC_FP8) return (d <= DEXP_E4M3) ? 1 : 0;   /* E4M3 : E5M2 */
    if (prec == PREC_FP6) return (d <= DEXP_E2M3) ? 1 : 0;   /* E2M3 : E3M2 */
    return 0;
}
'''

# ANCHOR must sit above mfp_encode/mfp_decode, which are the first
# consumers of the new helpers.
ANCHOR = '/* ================================================================\n *  MiniFP encode'


def apply_fastexp(h):
    old = '''    int   e;
    float fr = frexpf(av, &e);
    e--;  fr *= 2.0f;'''
    new = '''    int   e;
    float fr = fmant_of(av, e);   /* fr in [1,2), e = floor(log2 av) */'''
    assert old in h, 'frexpf block not found'
    h = h.replace(old, new)

    pairs = [
        ('        float mf = av / powf(2.0f, 1.0f - (float)bias) * (float)(1u << mb);',
         '        float mf = fscale2(av, bias - 1) * (float)(1u << mb);'),
        ('        v = ((float)m / (float)(1 << mb)) * powf(2.0f, 1.0f - (float)bias);',
         '        v = fscale2((float)m / (float)(1 << mb), 1 - bias);'),
        ('        v = (1.0f + (float)m / (float)(1 << mb)) * powf(2.0f, (float)(exp - bias));',
         '        v = fscale2(1.0f + (float)m / (float)(1 << mb), exp - bias);'),
    ]
    for old, new in pairs:
        assert old in h, 'powf site not found: ' + old[:50]
        h = h.replace(old, new)
    return h


def apply_sexp(h, c):
    old = '''inline ap_uint<34> pack_base_info(int prec, int row, int col) {
    ap_uint<34> h = 0;
    h(33, 32) = (ap_uint<2> )prec;
    h(31, 16) = (ap_uint<16>)row;
    h(15,  0) = (ap_uint<16>)col;
    return h;
}'''
    new = '''inline ap_uint<34> pack_base_info(int prec, int row_local, int sexp, int col) {
    ap_uint<34> h = 0;
    h(33, 32) = (ap_uint<2> )prec;
    h(31, 24) = (ap_uint<8> )row_local;   /* was 16-bit global row */
    h(23, 16) = (ap_uint<8> )sexp;        /* reclaimed: shared exp */
    h(15,  0) = (ap_uint<16>)col;
    return h;
}'''
    assert old in h, 'pack_base_info not found'
    h = h.replace(old, new)

    old = '''    int prec     = (int)(ap_uint<2> )pkt(127, 126);
    int row      = (int)(ap_uint<16>)pkt(125, 110);
    int base_col = (int)(ap_uint<16>)pkt(109,  94);'''
    new = '''    int prec     = (int)(ap_uint<2> )pkt(127, 126);
    int row      = (int)(ap_uint<8> )pkt(125, 118);   /* panel-local */
    int sexp     = (int)(ap_uint<8> )pkt(117, 110);   /* shared exp  */
    int base_col = (int)(ap_uint<16>)pkt(109,  94);
    const int sh = sexp - SEXP_BIAS;'''
    assert old in h, 'decoder header not found'
    h = h.replace(old, new)

    n = h.count('nv[e] = mfp_decode(enc, eb, mb, bias);')
    assert n == 3, 'decode sites: %d' % n
    h = h.replace('nv[e] = mfp_decode(enc, eb, mb, bias);',
                  'nv[e] = fscale2(mfp_decode(enc, eb, mb, bias), sh);')

    # ---- packer: grow tracking BOTH extremes, select tier by spread ----
    old = '''        int   min_c   = cols[i];
        float max_mag = absf_fast(vals[i]);
        int   count   = 1;

        EXTEND_BUF: for (int off = 1; off < NZ_MAX; off++) {
#pragma HLS UNROLL
            int j = i + off;
            if (j >= cnt) break;

            float new_mag  = absf_fast(vals[j]);
            float cand_max = (new_mag > max_mag) ? new_mag : max_mag;

            int p       = classify_prec(cand_max);
            int nz_lim  = get_nz_max(p);
            int max_off = get_max_offset(p);

            if (count >= nz_lim) break;
            if ((cols[j] - min_c) > max_off) break;

            max_mag = cand_max;
            count++;
        }

        int prec = classify_prec(max_mag);

        int fmt_val = 0, eb, mb, bias;
        if (prec == PREC_FP8) {
            fmt_val = fp8_select_pfmt(max_mag);
            get_fp8_params(fmt_val, eb, mb, bias);
        } else if (prec == PREC_FP6) {
            fmt_val = fp6_select_pfmt(max_mag);
            get_fp6_params(fmt_val, eb, mb, bias);
        } else {
            get_fp4_params(eb, mb, bias);
        }'''
    new = '''        int   min_c   = cols[i];
        float max_mag = absf_fast(vals[i]);
        float min_mag = max_mag;
        int   count   = 1;

        EXTEND_BUF: for (int off = 1; off < NZ_MAX; off++) {
#pragma HLS UNROLL
            int j = i + off;
            if (j >= cnt) break;

            float new_mag  = absf_fast(vals[j]);
            float cand_max = (new_mag > max_mag) ? new_mag : max_mag;
            float cand_min = (new_mag < min_mag && new_mag > 0.0f)
                             ? new_mag : min_mag;

            /* tier is set by the packet's SPREAD, not its magnitude */
            int p       = tier_for_range(cand_max, cand_min);
            int nz_lim  = get_nz_max(p);
            int max_off = get_max_offset(p);

            if (count >= nz_lim) break;
            if ((cols[j] - min_c) > max_off) break;

            max_mag = cand_max;
            min_mag = cand_min;
            count++;
        }

        int prec = tier_for_range(max_mag, min_mag);

        int fmt_val = subfmt_for_range(prec, max_mag, min_mag), eb, mb, bias;
        if (prec == PREC_FP8) {
            get_fp8_params(fmt_val, eb, mb, bias);
        } else if (prec == PREC_FP6) {
            get_fp6_params(fmt_val, eb, mb, bias);
        } else {
            get_fp4_params(eb, mb, bias);
        }'''
    assert old in c, 'packer extend loop not found'
    c = c.replace(old, new)

    old = '''        pkt128_t pkt = 0;
        pkt(127, 94) = pack_base_info(prec, row, min_c);'''
    new = '''        /* Per-packet shared exponent.  Place the packet's largest value
         * on the sub-format's ceiling so the whole group lands inside the
         * representable band; pure exponent arithmetic, no multiplier. */
        const int sh   = fexp_of(max_mag) - fmax_exp(prec, fmt_val);
        const int sexp = sh + SEXP_BIAS;
        const int nsh  = -sh;

        pkt128_t pkt = 0;
        pkt(127, 94) = pack_base_info(prec, row, sexp, min_c);'''
    assert old in c, 'packer header not found'
    c = c.replace(old, new)

    n = c.count('ap_uint<8> enc = mfp_encode(vals[i + e], eb, mb, bias);')
    assert n == 3, 'encode sites: %d' % n
    c = c.replace('ap_uint<8> enc = mfp_encode(vals[i + e], eb, mb, bias);',
                  'ap_uint<8> enc = mfp_encode(fscale2(vals[i + e], nsh), eb, mb, bias);')

    subs = [
        ('        int gr  = row_base + pe_id + r * NUM_PE;',
         '        int gr  = pe_id + r * NUM_PE;   /* panel-local row ID */'),
        # dispatch_route reads the row field straight out of the packet, so
        # it has to move to the narrowed 8-bit field too -- otherwise it
        # reads (row<<8)|sexp, fails the row_span test, and silently drops
        # every packet.
        ('        int row = (int)(ap_uint<16>)pkt(125, 110);\n'
         '        int lr  = row - row_base;',
         '        int row = (int)(ap_uint<8> )pkt(125, 118);  /* panel-local */\n'
         '        int lr  = row;'),
        ('                int lr    = nr_arr[e] - row_base;',
         '                int lr    = nr_arr[e];   /* panel-local */'),
    ]
    for old, new in subs:
        assert old in c, 'row_base site not found: ' + old[:40]
        c = c.replace(old, new)
    return h, c


BQ_HELPER = r'''
/* ================================================================
 *  FP8 E4M3 -> FP32, by field assembly (no pow, no DSP).
 *
 *  The dense operand S is a row-wise softmax, so every row is a
 *  probability distribution and the spread of row maxima across a
 *  real graph is only 2^3..2^5 -- far inside E4M3's 2^17 dynamic
 *  range.  One GLOBAL shared exponent therefore suffices for all of
 *  S, folded into the A value once per non-zero rather than per
 *  lane, so the inner loop carries no scaling cost at all.
 *
 *  Storing S at 8 bits is what widens the datapath: a 128-bit port
 *  beat carries 16 operands instead of 4.  With FP32 operands the
 *  kernel reaches 35-59% of peak MAC throughput on the ZCU104 while
 *  drawing 5.5-10.1% of the port's bandwidth, so the binding
 *  constraint is MAC lanes, not bytes.
 * ================================================================ */
inline float fp8e4m3_to_float(ap_uint<8> q) {
#pragma HLS INLINE
    ap_uint<1> s = q[7];
    ap_uint<4> e = q(6, 3);
    ap_uint<3> m = q(2, 0);
    union { float f; unsigned u; } cv;
    if (e == 0) {
        if (m == 0) return 0.0f;
        int sh2 = m[2] ? 0 : (m[1] ? 1 : 2);
        unsigned mm = ((unsigned)m << (sh2 + 1)) & 0x7u;
        unsigned ee = (unsigned)(127 - 16 - sh2);
        cv.u = ((unsigned)s << 31) | (ee << 23) | (mm << 20);
    } else {
        cv.u = ((unsigned)s << 31) | ((unsigned)(e + 111) << 23)
             | ((unsigned)m << 20);
    }
    return cv.f;
}
'''


def apply_bq(h, c):
    """Quantised dense operand + 16-lane MAC."""
    # ---- header: lane count and port element type ----
    old = '#define VEC_W  4\ntypedef ap_uint<32 * VEC_W> bvec_t;          /* 128-bit wide B word */'
    new = ('#define VEC_W  16\n'
           'typedef ap_uint<8 * VEC_W> bvec_t;           '
           '/* 128-bit beat = 16 FP8 operands */')
    assert old in h, 'VEC_W block not found'
    h = h.replace(old, new)
    h = h.replace(ANCHOR, BQ_HELPER + '\n' + ANCHOR, 1)

    # ---- cpp: B fetch unpacks 8-bit lanes ----
    old = '''                bvec_t wide = b_port[b_base_vec + jj];
                FILL_B_UNPACK: for (int v = 0; v < VEC_W; v++) {
#pragma HLS UNROLL
                    int idx = jj * VEC_W + v;
                    if (idx < cl) {
                        ap_uint<32> bits = (ap_uint<32>)wide(32 * v + 31, 32 * v);
                        b_line[idx] = bits_to_float(bits);
                    }
                }'''
    new = '''                bvec_t wide = b_port[b_base_vec + jj];
                FILL_B_UNPACK: for (int v = 0; v < VEC_W; v++) {
#pragma HLS UNROLL
                    int idx = jj * VEC_W + v;
                    if (idx < cl) {
                        ap_uint<8> q = (ap_uint<8>)wide(8 * v + 7, 8 * v);
                        b_line[idx] = fp8e4m3_to_float(q);
                    }
                }'''
    assert old in c, 'B unpack loop not found'
    c = c.replace(old, new)

    # ---- cpp: fold the global S exponent into the A value, once per nz ----
    old = '''        int   lr_pe = (int)nz_row_pe[e];
        int   ac    = (int)nz_col_k[e];
        float av    = nz_val[e];'''
    new = '''        int   lr_pe = (int)nz_row_pe[e];
        int   ac    = (int)nz_col_k[e];
        /* one scaling per non-zero, not per lane */
        float av    = fscale2(nz_val[e], b_sexp);'''
    assert old in c, 'nz unpack not found'
    c = c.replace(old, new)

    # ---- thread b_sexp down: cb_fill_phase -> PE -> top level ----
    subs = [
        ('    int         cl_vec,\n    int         rs_pe)',
         '    int         cl_vec,\n    int         rs_pe,\n    int         b_sexp)'),
        ('                      b_port, n_vec,\n'
         '                      cs, cl, cl_vec, rs_pe);',
         '                      b_port, n_vec,\n'
         '                      cs, cl, cl_vec, rs_pe, b_sexp);'),
        ('    int       local_k,\n    int       pe_id,\n'
         '    hls::stream<pkt128_t> &s_out)',
         '    int       local_k,\n    int       pe_id,\n    int       b_sexp,\n'
         '    hls::stream<pkt128_t> &s_out)'),
    ]
    for old, new in subs:
        assert old in c, 'bq thread site not found: ' + old[:48].replace('\n', '|')
        c = c.replace(old, new)

    # PE call sites: insert b_sexp before the output stream argument
    for i in range(4):
        old = ('K, N, row_span, tn_size, num_cb, local_k, %d, s_pkt%d);' % (i, i))
        if old in c:
            c = c.replace(old,
                          'K, N, row_span, tn_size, num_cb, local_k, %d, '
                          'b_sexp, s_pkt%d);' % (i, i))

    # top-level kernel argument + AXI-lite register
    old = '    int       top_k, int local_k);'
    assert old in h, 'kernel declaration not found'
    h = h.replace(old, '    int       top_k, int local_k,\n'
                       '    int       b_sexp);')
    old = '    int       top_k, int local_k)'
    assert old in c, 'kernel signature not found'
    c = c.replace(old, '    int       top_k, int local_k,\n    int       b_sexp)')
    old = '#pragma HLS INTERFACE s_axilite port=local_k'
    assert old in c, 'axilite local_k not found'
    c = c.replace(old, old + '\n#pragma HLS INTERFACE s_axilite port=b_sexp')

    # 16-way cyclic partitioning of c_buf puts 16 shallow banks in URAM,
    # which wastes most of each 36 KB block and overruns the 96 URAMs on a
    # ZU7EV.  The banks are 4 KB -- exactly a BRAM36 -- so bind them there.
    old = '#pragma HLS BIND_STORAGE variable=c_buf type=RAM_2P impl=uram'
    assert old in c, 'c_buf binding not found'
    c = c.replace(old, '#pragma HLS BIND_STORAGE variable=c_buf type=RAM_2P impl=bram')
    return h, c


RUN_TCL = '''open_component -reset cmp -flow_target vivado
set_part {%s}
create_clock -period %s -name default
add_files spmm_accel.cpp
add_files -cflags "-std=c++14" spmm_accel.h
set_top spmm_kernel
csynth_design
exit
'''


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', default=os.path.join(HERE, 'b0'),
                   help='directory holding B0 spmm_accel.{h,cpp} (default: hls/b0)')
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--part', default='xczu7ev-ffvc1156-2-e')
    p.add_argument('--period', default='10', help='HLS clock period, ns')
    p.add_argument('--fastexp', action='store_true')
    p.add_argument('--sexp', action='store_true')
    p.add_argument('--bq', action='store_true',
                   help='FP8 E4M3 dense operand + 16-lane MAC')
    # capacity knobs: maximum Top-k, rows per panel and column-block width
    p.add_argument('--topk', type=int, default=None)
    p.add_argument('--span', type=int, default=None)
    p.add_argument('--tn', type=int, default=None)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    h = io.open(os.path.join(a.base, 'spmm_accel.h'), encoding='utf-8').read()
    c = io.open(os.path.join(a.base, 'spmm_accel.cpp'), encoding='utf-8').read()

    if a.fastexp or a.sexp:
        assert ANCHOR in h, 'helper anchor not found'
        h = h.replace(ANCHOR, SEXP_HELPERS + '\n' + ANCHOR, 1)
    if a.fastexp:
        h = apply_fastexp(h)
    if a.sexp:
        h, c = apply_sexp(h, c)
    if a.bq:
        h, c = apply_bq(h, c)

    for flag, macro in ((a.topk, 'MAX_TOP_K'), (a.span, 'MAX_ROW_SPAN'),
                        (a.tn, 'MAX_TN_SIZE')):
        if flag is None:
            continue
        pat = '#define %s' % macro
        lines = h.split('\n')
        for i, ln in enumerate(lines):
            if ln.startswith(pat):
                lines[i] = '#define %-13s %d' % (macro, flag)
                break
        else:
            raise SystemExit('macro not found: ' + macro)
        h = '\n'.join(lines)
    # MAX_LOCAL_K tracks MAX_TOP_K
    if a.topk is not None:
        lines = h.split('\n')
        for i, ln in enumerate(lines):
            if ln.startswith('#define MAX_LOCAL_K'):
                lines[i] = '#define %-13s %d' % ('MAX_LOCAL_K', a.topk)
                break
        h = '\n'.join(lines)

    for name, text in (('spmm_accel.h', h), ('spmm_accel.cpp', c),
                       ('run.tcl', RUN_TCL % (a.part, a.period))):
        io.open(os.path.join(a.out, name), 'w', encoding='utf-8', newline='\n').write(text)
    print('generated %s: part=%s period=%s ns fastexp=%s sexp=%s bq=%s'
          % (a.out, a.part, a.period, a.fastexp, a.sexp, a.bq))


if __name__ == '__main__':
    main()
