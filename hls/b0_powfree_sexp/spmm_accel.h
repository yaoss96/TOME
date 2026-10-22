// ==================== spmm_accel.h ====================
#ifndef SPMM_ACCEL_H
#define SPMM_ACCEL_H

#include <ap_int.h>
#include <hls_stream.h>
#include <cmath>

/* ================================================================
 *  Types
 * ================================================================ */
typedef ap_uint<128> pkt128_t;
typedef ap_uint<64>  ltk_word_t;

/* ---- B-port vectorisation ---- */
#define VEC_W  4
typedef ap_uint<32 * VEC_W> bvec_t;          /* 128-bit wide B word */

/* ================================================================
 *  Constants
 * ================================================================ */
#define PREC_FP8  0
#define PREC_FP6  1
#define PREC_FP4  2
#define PREC_SENTINEL 3           /* end-of-stream marker (never in real data) */

#define NZ_FP8  5
#define NZ_FP6  6
#define NZ_FP4  7
#define NZ_MAX  7

/* ================================================================
 *  Representable-magnitude ceilings
 * ================================================================ */
#define FP8_REPR_MAX  1.75f
#define FP6_REPR_MAX  0.875f
#define FP4_REPR_MAX  0.013672f

#define NUM_PE    4

#define MAX_TOP_K     200
#define MAX_LOCAL_K   200

#define MAX_ROW_SPAN  256
#define MAX_TN_SIZE   256
#define MAX_A_PKTS    int(MAX_ROW_SPAN*MAX_TOP_K/5)
#define MAX_RES_PKTS  (MAX_ROW_SPAN * MAX_TOP_K)
#define RES_BUF_SIZE  (MAX_RES_PKTS + 1)

#define MAX_ROWS_PER_PE  ((MAX_ROW_SPAN + NUM_PE - 1) / NUM_PE)


/* ------------------------------------------------------------------
 *  Maximum decoded non-zeros stored per PE.
 * ------------------------------------------------------------------ */
#define MAX_NZ_PER_PE  (MAX_ROWS_PER_PE * MAX_TOP_K)   /* 64*256 = 16384 */

#define MAX_OUT_PKTS_PER_PE  (MAX_ROWS_PER_PE * MAX_LOCAL_K)

#define AXI_DEPTH_A_PACKETS  (65536*4)
#define AXI_DEPTH_B_ELEMS    (1048576*128)
#define AXI_DEPTH_B_VECS     (AXI_DEPTH_B_ELEMS / VEC_W)

#define MAX_OFF_FP8  1023
#define MAX_OFF_FP6   511
#define MAX_OFF_FP4   511

/* ================================================================
 *  Slot / offset helpers
 * ================================================================ */
inline int get_nz_max(int prec) {
    if (prec == PREC_FP8) return NZ_FP8;
    if (prec == PREC_FP6) return NZ_FP6;
    return NZ_FP4;
}

inline int get_max_offset(int prec) {
    if (prec == PREC_FP8) return MAX_OFF_FP8;
    if (prec == PREC_FP6) return MAX_OFF_FP6;
    return MAX_OFF_FP4;
}

/* ================================================================
 *  MFP format parameters
 * ================================================================ */
inline void get_fp8_params(int fmt, int &eb, int &mb, int &bias) {
    if (fmt == 0) { eb = 5; mb = 2; bias = 31; }
    else          { eb = 4; mb = 3; bias = 16; }
}

inline void get_fp6_params(int fmt, int &eb, int &mb, int &bias) {
    if (fmt == 0) { eb = 3; mb = 2; bias = 8; }
    else          { eb = 2; mb = 3; bias = 5; }
}

inline void get_fp4_params(int &eb, int &mb, int &bias) {
    eb = 1; mb = 2; bias = 8;
}

/* ================================================================
 *  Per-packet sub-format selection
 * ================================================================ */
inline int fp8_select_pfmt(float max_mag) {
    return (max_mag > 0.93f) ? 0 : 1;
}

inline int fp6_select_pfmt(float max_mag) {
    if (max_mag > 0.46f) return 0;
    if (max_mag < 0.06f) return 0;
    return 1;
}


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

/* ================================================================
 *  MiniFP encode
 * ================================================================ */
inline ap_uint<8> mfp_encode(float val, int eb, int mb, int bias) {
    if (val == 0.0f) return (ap_uint<8>)0;

    unsigned sgn = (val < 0.0f) ? 1u : 0u;
    float    av  = fabsf(val);

    unsigned exp_max  = (1u << eb) - 1u;
    unsigned mant_max = (1u << mb) - 1u;

    int   e;
    float fr = fmant_of(av, e);   /* fr in [1,2), e = floor(log2 av) */

    int eb_val = e + bias;

    ap_uint<8> r = 0;

    if (eb_val > (int)exp_max) {
        r = (sgn << (eb + mb)) | (exp_max << mb) | mant_max;
    } else if (eb_val <= 0) {
        float mf = fscale2(av, bias - 1) * (float)(1u << mb);
        unsigned m = (unsigned)(mf + 0.5f);
        if (m > mant_max) m = mant_max;
        r = (sgn << (eb + mb)) | m;
    } else {
        float mf = (fr - 1.0f) * (float)(1u << mb);
        unsigned m = (unsigned)(mf + 0.5f);
        if (m > mant_max) {
            m = 0; eb_val++;
            if (eb_val > (int)exp_max) {
                eb_val = (int)exp_max;
                m = mant_max;
            }
        }
        r = (sgn << (eb + mb)) | ((unsigned)eb_val << mb) | m;
    }
    return r;
}

/* ================================================================
 *  MiniFP decode
 * ================================================================ */
inline float mfp_decode(ap_uint<8> enc, int eb, int mb, int bias) {
    unsigned mant_max = (1u << mb) - 1u;

    int s   = ((int)(unsigned)enc >> (eb + mb)) & 1;
    int exp = ((int)(unsigned)enc >> mb) & ((1 << eb) - 1);
    int m   = (int)(unsigned)enc & (int)mant_max;

    if (exp == 0 && m == 0) return 0.0f;

    float v;
    if (exp == 0)
        v = fscale2((float)m / (float)(1 << mb), 1 - bias);
    else
        v = fscale2(1.0f + (float)m / (float)(1 << mb), exp - bias);

    return s ? -v : v;
}

/* ================================================================
 *  Pack 34-bit base-info
 * ================================================================ */
inline ap_uint<34> pack_base_info(int prec, int row_local, int sexp, int col) {
    ap_uint<34> h = 0;
    h(33, 32) = (ap_uint<2> )prec;
    h(31, 24) = (ap_uint<8> )row_local;   /* was 16-bit global row */
    h(23, 16) = (ap_uint<8> )sexp;        /* reclaimed: shared exp */
    h(15,  0) = (ap_uint<16>)col;
    return h;
}

/* ================================================================
 *  Decode a 128-bit packet
 * ================================================================ */
inline void decode_packet_inline(const pkt128_t &pkt,
                                 float *nv, int *nr, int *nc, bool *vd)
{
#pragma HLS INLINE

    int prec     = (int)(ap_uint<2> )pkt(127, 126);
    int row      = (int)(ap_uint<8> )pkt(125, 118);   /* panel-local */
    int sexp     = (int)(ap_uint<8> )pkt(117, 110);   /* shared exp  */
    int base_col = (int)(ap_uint<16>)pkt(109,  94);
    const int sh = sexp - SEXP_BIAS;

    DECODE_INIT: for (int e = 0; e < NZ_MAX; e++) {
#pragma HLS UNROLL
        nv[e] = 0.0f;
        nr[e] = 0;
        nc[e] = 0;
        vd[e] = false;
    }

    if (prec == PREC_FP8) {
        int fmt = (int)(ap_uint<1>)pkt(93, 93);
        int cnt = (int)(ap_uint<3>)pkt(92, 90);
        int eb, mb, bias;
        get_fp8_params(fmt, eb, mb, bias);

        DECODE_FP8: for (int e = 0; e < NZ_FP8; e++) {
#pragma HLS UNROLL
            if (e < cnt) {
                int b = 89 - e * 18;
                ap_uint<8> enc = (ap_uint<8>)pkt(b, b - 7);
                int coff = (int)(ap_uint<10>)pkt(b - 8, b - 17);
                nv[e] = fscale2(mfp_decode(enc, eb, mb, bias), sh);
                nr[e] = row;
                nc[e] = base_col + coff;
                vd[e] = true;
            }
        }

    } else if (prec == PREC_FP6) {
        int fmt = (int)(ap_uint<1>)pkt(93, 93);
        int cnt = (int)(ap_uint<3>)pkt(92, 90);
        int eb, mb, bias;
        get_fp6_params(fmt, eb, mb, bias);

        DECODE_FP6: for (int e = 0; e < NZ_FP6; e++) {
#pragma HLS UNROLL
            if (e < cnt) {
                int b = 89 - e * 15;
                ap_uint<8> enc = 0;
                enc(5, 0) = (ap_uint<6>)pkt(b, b - 5);
                int coff = (int)(ap_uint<9>)pkt(b - 6, b - 14);
                nv[e] = fscale2(mfp_decode(enc, eb, mb, bias), sh);
                nr[e] = row;
                nc[e] = base_col + coff;
                vd[e] = true;
            }
        }

    } else {
        int cnt = (int)(ap_uint<3>)pkt(93, 91);
        int eb, mb, bias;
        get_fp4_params(eb, mb, bias);

        DECODE_FP4: for (int e = 0; e < NZ_FP4; e++) {
#pragma HLS UNROLL
            if (e < cnt) {
                int b = 90 - e * 13;
                ap_uint<8> enc = 0;
                enc(3, 0) = (ap_uint<4>)pkt(b, b - 3);
                int coff = (int)(ap_uint<9>)pkt(b - 4, b - 12);
                nv[e] = fscale2(mfp_decode(enc, eb, mb, bias), sh);
                nr[e] = row;
                nc[e] = base_col + coff;
                vd[e] = true;
            }
        }
    }
}

/* ================================================================
 *  Range-based precision classification
 * ================================================================ */
inline int classify_prec(float mag) {
    if (mag <= FP4_REPR_MAX) return PREC_FP4;
    if (mag <= FP6_REPR_MAX) return PREC_FP6;
    return PREC_FP8;
}

/* ================================================================
 *  Top-level kernel declaration
 *
 *  NOTE: N and tn_size must be multiples of VEC_W (4).
 * ================================================================ */
#ifdef __cplusplus
extern "C" {
#endif

void spmm_kernel(
    pkt128_t *a_packets,
    bvec_t   *b_port0,
    bvec_t   *b_port1,
    bvec_t   *b_port2,
    bvec_t   *b_port3,
    pkt128_t *result_packets,
    int       total_pkts,
    int       row_base,
    int       M, int K, int N,
    int       row_span, int tn_size,
    int       top_k, int local_k);

#ifdef __cplusplus
}
#endif

#endif /* SPMM_ACCEL_H */