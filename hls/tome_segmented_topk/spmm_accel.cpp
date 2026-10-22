// ==================== spmm_accel.cpp ====================
#include "spmm_accel.h"
#include <cstring>
#include <cmath>

/* ================================================================
 *  Helpers: float <-> ap_uint<32> bit-level conversion
 * ================================================================ */
static ap_uint<32> float_to_bits(float v) {
#pragma HLS INLINE
    union { float f; unsigned int u; } conv;
    conv.f = v;
    return (ap_uint<32>)conv.u;
}

static float bits_to_float(ap_uint<32> b) {
#pragma HLS INLINE
    union { float f; unsigned int u; } conv;
    conv.u = (unsigned int)b;
    return conv.f;
}

/* ================================================================
 *  topk_insert
 * ================================================================ */
static void topk_insert(float *tk_v, int *tk_c, int &tk_cnt,
                         float &tk_min_v, int &tk_min_i,
                         float *seg_min, int *seg_idx,
                         int tk, float val, int col)
{
#pragma HLS INLINE
    float aval = fabsf(val);
    if (aval < 1e-12f) return;

    if (tk_cnt < tk) {
        int p = tk_cnt;
        tk_v[p] = val;
        tk_c[p] = col;
        int s = p / TK_SEG;
        if (aval < seg_min[s]) { seg_min[s] = aval; seg_idx[s] = p; }
        if (tk_cnt == 0 || aval < tk_min_v) { tk_min_v = aval; tk_min_i = p; }
        tk_cnt++;
    } else {
        if (aval <= tk_min_v) return;

        int p = tk_min_i;
        tk_v[p] = val;
        tk_c[p] = col;

        /* exactly one segment is dirty */
        int   s  = p / TK_SEG;
        int   b  = s * TK_SEG;
        float sm = 3.4e38f;
        int   si = b;
        SEG_RESCAN: for (int t = 0; t < TK_SEG; t++) {
#pragma HLS UNROLL
            int q = b + t;
            if (q < tk) {
                float qa = fabsf(tk_v[q]);
                if (qa < sm) { sm = qa; si = q; }
            }
        }
        seg_min[s] = sm;
        seg_idx[s] = si;

        /* reduce the segment minima */
        float gm = 3.4e38f;
        int   gi = 0;
        SEG_REDUCE: for (int t = 0; t < TK_NSEG; t++) {
#pragma HLS UNROLL
            if (t * TK_SEG < tk && seg_min[t] < gm) {
                gm = seg_min[t];
                gi = seg_idx[t];
            }
        }
        tk_min_v = gm;
        tk_min_i = gi;
    }
}

static float absf_fast(float v) {
#pragma HLS INLINE
    ap_uint<32> b = float_to_bits(v);
    b[31] = 0;
    return bits_to_float(b);
}
// ==================== Optimised packing (replaces the original sort_by_col, greedy_pack_to_buf and Phase 5) ====================

/* ================================================================
 *  sort_by_col  - bottom-up merge sort  O(n log n)
 *
 *  Replaces the original O(n^2) selection sort.
 *  At cnt = 200: 8 passes x (200x2 merge + 200x1 copy) ~ 4,800 cycles,
 *  vs. ~40,000 cycles for the selection sort, about 8x faster.
 * ================================================================ */
static void sort_by_col(float *vals, int *cols, int cnt)
{
#pragma HLS INLINE off
    if (cnt <= 1) return;

    float tmp_v[MAX_LOCAL_K];
    int   tmp_c[MAX_LOCAL_K];
#pragma HLS BIND_STORAGE variable=tmp_v type=RAM_2P impl=lutram
#pragma HLS BIND_STORAGE variable=tmp_c type=RAM_2P impl=lutram

    int width = 1;
    SORT_PASS: for (int pass = 0; pass < 8; pass++) {
#pragma HLS LOOP_TRIPCOUNT max=8
        if (width >= cnt) break;

        SORT_BLOCK: for (int blk = 0; blk < cnt; blk += 2 * width) {
#pragma HLS LOOP_TRIPCOUNT max=MAX_LOCAL_K
            int left  = blk;
            int mid   = left + width;
            int right = left + 2 * width;
            if (mid   > cnt) mid   = cnt;
            if (right > cnt) right = cnt;

            int l = left, r = mid;

            SORT_ELEM: for (int d = left; d < right; d++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=MAX_LOCAL_K
                bool pick_left;
                if      (l >= mid)   pick_left = false;
                else if (r >= right) pick_left = true;
                else                 pick_left = (cols[l] <= cols[r]);

                tmp_v[d] = pick_left ? vals[l] : vals[r];
                tmp_c[d] = pick_left ? cols[l] : cols[r];
                l += pick_left ? 1 : 0;
                r += pick_left ? 0 : 1;
            }
        }

        SORT_COPYBACK: for (int i = 0; i < cnt; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=MAX_LOCAL_K
            vals[i] = tmp_v[i];
            cols[i] = tmp_c[i];
        }

        width <<= 1;
    }
}

/* ================================================================
 *  greedy_pack_to_buf  - optimised version
 *
 *  Changes:
 *    1. while -> bounded for loop (HLS can estimate the trip count)
 *    2. columns are already sorted, so min_c = cols[i] and
 *       offset = cols[j] - cols[i]; no min_c / max_c tracking
 *    3. no redundant classify_prec call after EXTEND_BUF
 *    4. the npkt overflow check moves to the loop entry
 * ================================================================ */
static int greedy_pack_to_buf(
    int       row,
    float    *vals,
    int      *cols,
    int       cnt,
    pkt128_t *pkt_buf,
    int       npkt,
    int       max_pkts)
{
#pragma HLS INLINE off

    int i = 0;

    GREEDY_BUF: for (int iter = 0; iter < MAX_LOCAL_K; iter++) {
#pragma HLS LOOP_TRIPCOUNT max=MAX_LOCAL_K
        if (i >= cnt || npkt >= max_pkts) break;

        int   min_c   = cols[i];
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
        }

        /* Per-packet shared exponent.  Place the packet's largest value
         * on the sub-format's ceiling so the whole group lands inside the
         * representable band; pure exponent arithmetic, no multiplier. */
        const int sh   = fexp_of(max_mag) - fmax_exp(prec, fmt_val);
        const int sexp = sh + SEXP_BIAS;
        const int nsh  = -sh;

        pkt128_t pkt = 0;
        pkt(127, 94) = pack_base_info(prec, row, sexp, min_c);

        if (prec == PREC_FP8) {
            pkt(93, 93) = (ap_uint<1>)fmt_val;
            pkt(92, 90) = (ap_uint<3>)count;
            PACK_FP8_BUF: for (int e = 0; e < NZ_FP8; e++) {
#pragma HLS UNROLL
                if (e < count) {
                    int lc = cols[i + e] - min_c;
                    ap_uint<8> enc = mfp_encode(fscale2(vals[i + e], nsh), eb, mb, bias);
                    int bb = 89 - e * 18;
                    pkt(bb, bb - 7)      = enc(7, 0);
                    pkt(bb - 8, bb - 17) = (ap_uint<10>)(lc & 0x3FF);
                }
            }
        } else if (prec == PREC_FP6) {
            pkt(93, 93) = (ap_uint<1>)fmt_val;
            pkt(92, 90) = (ap_uint<3>)count;
            PACK_FP6_BUF: for (int e = 0; e < NZ_FP6; e++) {
#pragma HLS UNROLL
                if (e < count) {
                    int lc = cols[i + e] - min_c;
                    ap_uint<8> enc = mfp_encode(fscale2(vals[i + e], nsh), eb, mb, bias);
                    int bb = 89 - e * 15;
                    pkt(bb, bb - 5)      = (ap_uint<6>)enc(5, 0);
                    pkt(bb - 6, bb - 14) = (ap_uint<9>)(lc & 0x1FF);
                }
            }
        } else {
            pkt(93, 91) = (ap_uint<3>)count;
            PACK_FP4_BUF: for (int e = 0; e < NZ_FP4; e++) {
#pragma HLS UNROLL
                if (e < count) {
                    int lc = cols[i + e] - min_c;
                    ap_uint<8> enc = mfp_encode(fscale2(vals[i + e], nsh), eb, mb, bias);
                    int bb = 90 - e * 13;
                    pkt(bb, bb - 3)      = (ap_uint<4>)enc(3, 0);
                    pkt(bb - 4, bb - 12) = (ap_uint<9>)(lc & 0x1FF);
                }
            }
        }

        pkt_buf[npkt++] = pkt;
        i += count;
    }

    return npkt;
}

/* ================================================================
 *  sort_nz_by_col  -  bottom-up merge sort on NZ column key
 * ================================================================ */
static void sort_nz_by_col(
    float       *nz_val,
    ap_uint<8>  *nz_row_pe,
    ap_uint<16> *nz_col_k,
    int          n)
{
#pragma HLS INLINE off
    if (n <= 1) return;

    float       tmp_v[MAX_NZ_PER_PE];
    ap_uint<8>  tmp_r[MAX_NZ_PER_PE];
    ap_uint<16> tmp_c[MAX_NZ_PER_PE];
#pragma HLS BIND_STORAGE variable=tmp_v type=RAM_2P impl=uram
#pragma HLS BIND_STORAGE variable=tmp_r type=RAM_2P impl=bram
#pragma HLS BIND_STORAGE variable=tmp_c type=RAM_2P impl=bram

    int width = 1;
    MERGE_PASS: for (int pass = 0; pass < 15; pass++) {
#pragma HLS LOOP_TRIPCOUNT max=15
        if (width >= n) break;

        MERGE_BLOCK: for (int blk = 0; blk < n; blk += 2 * width) {
#pragma HLS LOOP_TRIPCOUNT max=MAX_NZ_PER_PE
            int left  = blk;
            int mid   = left + width;
            int right = left + 2 * width;
            if (mid   > n) mid   = n;
            if (right > n) right = n;

            int l = left, r = mid;

            MERGE_ELEM: for (int d = left; d < right; d++) {
#pragma HLS PIPELINE II=2
#pragma HLS LOOP_TRIPCOUNT max=MAX_NZ_PER_PE
                float       v_l = nz_val[l];
                float       v_r = nz_val[r];
                ap_uint<8>  r_l = nz_row_pe[l];
                ap_uint<8>  r_r = nz_row_pe[r];
                ap_uint<16> c_l = nz_col_k[l];
                ap_uint<16> c_r = nz_col_k[r];

                bool pick_left;
                if (l >= mid)        pick_left = false;
                else if (r >= right) pick_left = true;
                else                 pick_left = ((unsigned)c_l <= (unsigned)c_r);

                tmp_v[d] = pick_left ? v_l : v_r;
                tmp_r[d] = pick_left ? r_l : r_r;
                tmp_c[d] = pick_left ? c_l : c_r;
                l += pick_left ? 1 : 0;
                r += pick_left ? 0 : 1;
            }
        }

        COPY_BACK: for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=MAX_NZ_PER_PE
            nz_val[i]    = tmp_v[i];
            nz_row_pe[i] = tmp_r[i];
            nz_col_k[i]  = tmp_c[i];
        }

        width <<= 1;
    }
}

/* ================================================================
 *  CB_LOOP fill phase  (NEW)
 *
 *  Clears c_buf, then runs the A-stationary MAC with wide-B
 *  caching.  Paired with cb_drain_phase inside a DATAFLOW loop
 *  so that HLS creates a PIPO (ping-pong) double-buffer for
 *  c_buf:  fill[N+1] overlaps with drain[N].
 * ================================================================ */
static void cb_fill_phase(
    float       c_buf[MAX_ROWS_PER_PE][MAX_TN_SIZE],
    float       nz_val[MAX_NZ_PER_PE],
    ap_uint<8>  nz_row_pe[MAX_NZ_PER_PE],
    ap_uint<16> nz_col_k[MAX_NZ_PER_PE],
    int         num_nz,
    bvec_t     *b_port,
    int         n_vec,
    int         cs,
    int         cl,
    int         cl_vec,
    int         rs_pe,
    int         b_sexp)
{
#pragma HLS INLINE off

    float b_line[MAX_TN_SIZE];
#pragma HLS ARRAY_PARTITION variable=b_line cyclic factor=VEC_W

    /* ---------- vectorised clear ---------- */
    FILL_CLEAR_C: for (int r = 0; r < rs_pe; r++) {
#pragma HLS LOOP_TRIPCOUNT max=MAX_ROWS_PER_PE
        FILL_CLEAR_C_VEC: for (int jj = 0; jj < cl_vec; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=(MAX_TN_SIZE/VEC_W)
            FILL_CLEAR_C_LANE: for (int v = 0; v < VEC_W; v++) {
#pragma HLS UNROLL
                int idx = jj * VEC_W + v;
                if (idx < cl) c_buf[r][idx] = 0.0f;
            }
        }
    }

    /* ---------- A-stationary MAC with wide B caching ---------- */
    int b_line_col = -1;

    FILL_NZ_LOOP: for (int e = 0; e < num_nz; e++) {
#pragma HLS LOOP_TRIPCOUNT max=MAX_NZ_PER_PE
        int   lr_pe = (int)nz_row_pe[e];
        int   ac    = (int)nz_col_k[e];
        /* one scaling per non-zero, not per lane */
        float av    = fscale2(nz_val[e], b_sexp);

        /* load B row segment only on column change (VEC_W-wide) */
        if (ac != b_line_col) {
            long long b_base_vec = (long long)ac * n_vec + (cs / VEC_W);
            FILL_B_READ: for (int jj = 0; jj < cl_vec; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=(MAX_TN_SIZE/VEC_W)
                bvec_t wide = b_port[b_base_vec + jj];
                FILL_B_UNPACK: for (int v = 0; v < VEC_W; v++) {
#pragma HLS UNROLL
                    int idx = jj * VEC_W + v;
                    if (idx < cl) {
                        ap_uint<8> q = (ap_uint<8>)wide(8 * v + 7, 8 * v);
                        b_line[idx] = fp8e4m3_to_float(q);
                    }
                }
            }
            b_line_col = ac;
        }

        /* VEC_W-wide MAC */
        FILL_MAC_LOOP: for (int jj = 0; jj < cl_vec; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=(MAX_TN_SIZE/VEC_W)
            FILL_MAC_LANE: for (int v = 0; v < VEC_W; v++) {
#pragma HLS UNROLL
                int idx = jj * VEC_W + v;
                if (idx < cl) {
                    c_buf[lr_pe][idx] += av * b_line[idx];
                }
            }
        }
    }
}

/* ================================================================
 *  CB_LOOP drain phase  (NEW)
 *
 *  Reads the completed c_buf and inserts each column-block
 *  result into the per-row top-k heap.
 * ================================================================ */
static void cb_drain_phase(
    float c_buf[MAX_ROWS_PER_PE][MAX_TN_SIZE],
    float tk_val[MAX_ROWS_PER_PE][MAX_LOCAL_K],
    int   tk_col[MAX_ROWS_PER_PE][MAX_LOCAL_K],
    int   tk_cnt[MAX_ROWS_PER_PE],
    float tk_min_val[MAX_ROWS_PER_PE],
    int   tk_min_idx[MAX_ROWS_PER_PE],
    float seg_min[MAX_ROWS_PER_PE][TK_NSEG],
    int   seg_idx[MAX_ROWS_PER_PE][TK_NSEG],
    int   rs_pe,
    int   cl,
    int   cs,
    int   lk)
{
#pragma HLS INLINE off

    DRAIN_TK_INSERT: for (int r = 0; r < rs_pe; r++) {
#pragma HLS LOOP_TRIPCOUNT max=MAX_ROWS_PER_PE
        DRAIN_TK_INSERT_J: for (int j = 0; j < cl; j++) {
#pragma HLS LOOP_TRIPCOUNT max=MAX_TN_SIZE
            topk_insert(tk_val[r], tk_col[r], tk_cnt[r],
                        tk_min_val[r], tk_min_idx[r],
                        seg_min[r], seg_idx[r],
                        lk, c_buf[r][j], cs + j);
        }
    }
}

/* ================================================================
 *  DATAFLOW stage 1 - dispatch_route
 * ================================================================ */
void dispatch_route(
    pkt128_t *a_packets,
    int       total_pkts,
    int       row_base,
    int       row_span,
    hls::stream<pkt128_t> &s_pe0,
    hls::stream<pkt128_t> &s_pe1,
    hls::stream<pkt128_t> &s_pe2,
    hls::stream<pkt128_t> &s_pe3)
{
    ROUTE: for (int i = 0; i < total_pkts; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=MAX_A_PKTS
        pkt128_t pkt = a_packets[i];
        int row = (int)(ap_uint<8> )pkt(125, 118);  /* panel-local */
        int lr  = row;
        if (lr >= 0 && lr < row_span) {
            int pe = lr & (NUM_PE - 1);
            if      (pe == 0) s_pe0.write(pkt);
            else if (pe == 1) s_pe1.write(pkt);
            else if (pe == 2) s_pe2.write(pkt);
            else              s_pe3.write(pkt);
        }
    }

    /* End-of-stream sentinel to every PE */
    pkt128_t sentinel = 0;
    sentinel(127, 126) = (ap_uint<2>)PREC_SENTINEL;
    s_pe0.write(sentinel);
    s_pe1.write(sentinel);
    s_pe2.write(sentinel);
    s_pe3.write(sentinel);
}

/* ================================================================
 *  DATAFLOW stage 2 - pe_spmm_compute_topk  (OPTIMISED v4)
 *
 *  v4 changes over v3:
 *    CB_LOOP is now a DATAFLOW loop.  c_buf is declared inside
 *    the loop body so that HLS creates a PIPO (ping-pong)
 *    double-buffer automatically:
 *        cb_fill_phase[N+1]  runs in parallel with
 *        cb_drain_phase[N]
 *    halving the idle time between column-block iterations.
 * ================================================================ */
void pe_spmm_compute_topk(
    hls::stream<pkt128_t> &s_in,
    bvec_t   *b_port,
    int       row_base,
    int       nrows_act,
    int       K, int N,
    int       row_span,
    int       tn_size,
    int       num_cb,
    int       local_k,
    int       pe_id,
    int       b_sexp,
    hls::stream<pkt128_t> &s_out)
{
    const int rs    = (nrows_act  > MAX_ROW_SPAN) ? MAX_ROW_SPAN : nrows_act;
    const int lk    = (local_k    > MAX_LOCAL_K)  ? MAX_LOCAL_K  : local_k;
    const int tn    = (tn_size    > MAX_TN_SIZE)  ? MAX_TN_SIZE  : tn_size;
    const int rs_pe = (rs > pe_id) ? ((rs - pe_id + NUM_PE - 1) / NUM_PE) : 0;
    const int n_vec = N / VEC_W;

    /* ============================================================
     *  Phase 1 - Stream-read + decode in ONE pass (no a_local)
     * ============================================================ */
    float       nz_val[MAX_NZ_PER_PE];
    ap_uint<8>  nz_row_pe[MAX_NZ_PER_PE];
    ap_uint<16> nz_col_k[MAX_NZ_PER_PE];
#pragma HLS BIND_STORAGE variable=nz_val    type=RAM_2P impl=uram
#pragma HLS BIND_STORAGE variable=nz_row_pe type=RAM_2P impl=lutram
#pragma HLS BIND_STORAGE variable=nz_col_k  type=RAM_2P impl=lutram

    int num_nz = 0;

    READ_AND_DECODE: for (int i = 0; i < MAX_A_PKTS + 1; i++) {
#pragma HLS LOOP_TRIPCOUNT max=MAX_A_PKTS
        pkt128_t pkt = s_in.read();

        /* sentinel check */
        if ((int)(ap_uint<2>)pkt(127, 126) == PREC_SENTINEL) break;

        float nv[NZ_MAX];
        int   nr_arr[NZ_MAX], nc_arr[NZ_MAX];
        bool  vd[NZ_MAX];
#pragma HLS ARRAY_PARTITION variable=nv      complete
#pragma HLS ARRAY_PARTITION variable=nr_arr  complete
#pragma HLS ARRAY_PARTITION variable=nc_arr  complete
#pragma HLS ARRAY_PARTITION variable=vd      complete

        decode_packet_inline(pkt, nv, nr_arr, nc_arr, vd);

        STORE_NZ: for (int e = 0; e < NZ_MAX; e++) {
            if (vd[e] && num_nz < MAX_NZ_PER_PE) {
                int lr    = nr_arr[e];   /* panel-local */
                int lr_pe = lr / NUM_PE;
                if (lr_pe >= 0 && lr_pe < rs_pe) {
                    nz_val[num_nz]    = nv[e];
                    nz_row_pe[num_nz] = (ap_uint<8>)lr_pe;
                    nz_col_k[num_nz]  = (ap_uint<16>)nc_arr[e];
                    num_nz++;
                }
            }
        }
    }

    /* ============================================================
     *  Phase 2 - Sort NZ by column (merge sort, O(n log n))
     * ============================================================ */
    sort_nz_by_col(nz_val, nz_row_pe, nz_col_k, num_nz);

    /* ============================================================
     *  Phase 3 - Initialise top-k buffers + per-row min tracking
     * ============================================================ */
    float tk_val[MAX_ROWS_PER_PE][MAX_LOCAL_K];
    int   tk_col[MAX_ROWS_PER_PE][MAX_LOCAL_K];
    int   tk_cnt[MAX_ROWS_PER_PE];
#pragma HLS BIND_STORAGE variable=tk_val type=RAM_2P impl=uram
#pragma HLS ARRAY_PARTITION variable=tk_val cyclic factor=TK_SEG dim=2
#pragma HLS BIND_STORAGE variable=tk_col type=RAM_2P impl=uram
#pragma HLS BIND_STORAGE variable=tk_cnt type=RAM_2P impl=lutram

    float tk_seg_min[MAX_ROWS_PER_PE][TK_NSEG];
    int   tk_seg_idx[MAX_ROWS_PER_PE][TK_NSEG];
#pragma HLS ARRAY_PARTITION variable=tk_seg_min complete dim=2
#pragma HLS ARRAY_PARTITION variable=tk_seg_idx complete dim=2
    float tk_min_val[MAX_ROWS_PER_PE];
    int   tk_min_idx[MAX_ROWS_PER_PE];
#pragma HLS BIND_STORAGE variable=tk_min_val type=RAM_1P impl=lutram
#pragma HLS BIND_STORAGE variable=tk_min_idx type=RAM_1P impl=lutram

    INIT_TK: for (int r = 0; r < rs_pe; r++) {
        tk_cnt[r]     = 0;
        tk_min_val[r] = 0.0f;
        INIT_SEG: for (int s = 0; s < TK_NSEG; s++) {
#pragma HLS UNROLL
            tk_seg_min[r][s] = 3.4e38f;
            tk_seg_idx[r][s] = s * TK_SEG;
        }
        tk_min_idx[r] = 0;
        INIT_TK_INNER: for (int k = 0; k < lk; k++) {
#pragma HLS PIPELINE II=1
            tk_val[r][k] = 0.0f;
            tk_col[r][k] = -1;
        }
    }

    /* ============================================================
     *  Phase 4 - A-element-stationary SpMM  (DATAFLOW double-buf)
     *
     *  CB_LOOP is a DATAFLOW loop.  c_buf is declared inside the
     *  loop body so HLS automatically creates a PIPO double-
     *  buffer, allowing cb_fill_phase of iteration N+1 to overlap
     *  with cb_drain_phase of iteration N.
     *
     *  Pipeline view (steady state):
     *
     *   fill:  |--N--|--N+1--|--N+2--|
     *   drain:       |--N---|--N+1--|--N+2--|
     *
     *  nz_b_port are read-only -> shared across fill calls.
     *  tk_* carry loop state    -> drain calls are serialised.
     * ============================================================ */
    CB_LOOP: for (int cb = 0; cb < num_cb; cb++) {
#pragma HLS DATAFLOW

        const int cs     = cb * tn;
        int ce = cs + tn;
        if (ce > N) ce = N;
        const int cl     = ce - cs;
        const int cl_vec = (cl + VEC_W - 1) / VEC_W;

        /* c_buf is local to the DATAFLOW region -> PIPO candidate */
        float c_buf[MAX_ROWS_PER_PE][MAX_TN_SIZE];
#pragma HLS BIND_STORAGE variable=c_buf type=RAM_2P impl=bram
#pragma HLS ARRAY_PARTITION variable=c_buf cyclic factor=VEC_W dim=2

        cb_fill_phase(c_buf,
                      nz_val, nz_row_pe, nz_col_k, num_nz,
                      b_port, n_vec,
                      cs, cl, cl_vec, rs_pe, b_sexp);

        cb_drain_phase(c_buf,
                       tk_val, tk_col, tk_cnt,
                       tk_min_val, tk_min_idx,
                       tk_seg_min, tk_seg_idx,
                       rs_pe, cl, cs, lk);
    }

    /* ============================================================
     *  Phase 5 - Sort + greedy-pack inside the PE
     *
     *  Each row's top-k results are sorted by column and greedily packed;
     *  COPY_TK fuses copy and zero-padding with a conditional assignment.
     * ============================================================ */
    pkt128_t pkt_buf[MAX_OUT_PKTS_PER_PE];
#pragma HLS BIND_STORAGE variable=pkt_buf type=RAM_2P impl=uram

    int npkt_total = 0;

    PACK_ROWS: for (int r = 0; r < rs_pe; r++) {
#pragma HLS LOOP_TRIPCOUNT max=MAX_ROWS_PER_PE

        int gr  = pe_id + r * NUM_PE;   /* panel-local row ID */
        int cnt = tk_cnt[r];
        if (cnt > lk) cnt = lk;
        if (cnt <= 0) continue;

        float rv[MAX_LOCAL_K];
        int   rc[MAX_LOCAL_K];
#pragma HLS BIND_STORAGE variable=rv type=RAM_2P impl=lutram
#pragma HLS BIND_STORAGE variable=rc type=RAM_2P impl=lutram

        COPY_TK: for (int k = 0; k < cnt; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=MAX_LOCAL_K
            rv[k] = tk_val[r][k];
            rc[k] = tk_col[r][k];
        }

        sort_by_col(rv, rc, cnt);

        npkt_total = greedy_pack_to_buf(gr, rv, rc, cnt,
                                        pkt_buf, npkt_total,
                                        MAX_OUT_PKTS_PER_PE);
    }

    pkt128_t hdr = 0;
    hdr(31, 0) = (ap_uint<32>)npkt_total;
    s_out.write(hdr);

    OUTPUT_PKTS: for (int i = 0; i < npkt_total; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=MAX_OUT_PKTS_PER_PE
        s_out.write(pkt_buf[i]);
    }
}

/* ================================================================
 *  DATAFLOW stage 3 - collect_write
 * ================================================================ */
static int drain_pe_stream(
    hls::stream<pkt128_t> &s_in,
    pkt128_t *result_packets,
    int       rpkt_idx)
{
#pragma HLS INLINE off

    pkt128_t hdr = s_in.read();
    int npkt = (int)(ap_uint<32>)hdr(31, 0);

    DRAIN_PKTS: for (int i = 0; i < npkt; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=MAX_OUT_PKTS_PER_PE
        if (rpkt_idx < MAX_RES_PKTS) {
            result_packets[rpkt_idx] = s_in.read();
            rpkt_idx++;
        } else {
            pkt128_t discard = s_in.read();
            (void)discard;
        }
    }
    return rpkt_idx;
}

void collect_write(
    hls::stream<pkt128_t> &s_pkt0,
    hls::stream<pkt128_t> &s_pkt1,
    hls::stream<pkt128_t> &s_pkt2,
    hls::stream<pkt128_t> &s_pkt3,
    pkt128_t *result_packets,
    int       row_base,
    int       nrows_act,
    int       row_span,
    int       local_k,
    int       top_k)
{
    int rpkt_idx = 1;

    rpkt_idx = drain_pe_stream(s_pkt0, result_packets, rpkt_idx);
    rpkt_idx = drain_pe_stream(s_pkt1, result_packets, rpkt_idx);
    rpkt_idx = drain_pe_stream(s_pkt2, result_packets, rpkt_idx);
    rpkt_idx = drain_pe_stream(s_pkt3, result_packets, rpkt_idx);

    pkt128_t hdr = 0;
    hdr(31, 0) = (ap_uint<32>)rpkt_idx;
    result_packets[0] = hdr;
}

/* ============================================================
 *  Top-level kernel (DATAFLOW)
 * ============================================================ */
extern "C" {
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
    int       top_k, int local_k,
    int       b_sexp)
{
#pragma HLS INTERFACE m_axi port=a_packets       offset=slave bundle=gmem0 \
    depth=AXI_DEPTH_A_PACKETS max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=b_port0         offset=slave bundle=gmem2 \
    depth=AXI_DEPTH_B_VECS max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=b_port1         offset=slave bundle=gmem3 \
    depth=AXI_DEPTH_B_VECS max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=b_port2         offset=slave bundle=gmem4 \
    depth=AXI_DEPTH_B_VECS max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=b_port3         offset=slave bundle=gmem5 \
    depth=AXI_DEPTH_B_VECS max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=result_packets  offset=slave bundle=gmem6 \
    depth=RES_BUF_SIZE max_read_burst_length=64 max_write_burst_length=64

#pragma HLS INTERFACE s_axilite port=total_pkts
#pragma HLS INTERFACE s_axilite port=row_base
#pragma HLS INTERFACE s_axilite port=M
#pragma HLS INTERFACE s_axilite port=K
#pragma HLS INTERFACE s_axilite port=N
#pragma HLS INTERFACE s_axilite port=row_span
#pragma HLS INTERFACE s_axilite port=tn_size
#pragma HLS INTERFACE s_axilite port=top_k
#pragma HLS INTERFACE s_axilite port=local_k
#pragma HLS INTERFACE s_axilite port=b_sexp
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS ALLOCATION function instances=pe_spmm_compute_topk limit=4

    if (row_span   < 1)             row_span   = 1;
    if (row_span   > MAX_ROW_SPAN)  row_span   = MAX_ROW_SPAN;
    if (tn_size    < 1)             tn_size    = 1;
    if (tn_size    > MAX_TN_SIZE)   tn_size    = MAX_TN_SIZE;
    if (total_pkts < 0)             total_pkts = 0;
    if (total_pkts > MAX_A_PKTS)    total_pkts = MAX_A_PKTS;
    if (top_k      < 1)             top_k      = 1;
    if (top_k      > MAX_TOP_K)     top_k      = MAX_TOP_K;
    if (local_k    < 1)             local_k    = 1;
    if (local_k    > MAX_LOCAL_K)   local_k    = MAX_LOCAL_K;

    /* Align tn_size to VEC_W */
    tn_size = (tn_size / VEC_W) * VEC_W;
    if (tn_size < VEC_W) tn_size = VEC_W;

    int nrows_act = (row_base < M)
        ? (((M - row_base) < row_span) ? (M - row_base) : row_span)
        : 0;
    if (nrows_act < 0) nrows_act = 0;

    int num_cb = (N > 0) ? ((N + tn_size - 1) / tn_size) : 0;

    hls::stream<pkt128_t> s_pe0("s_pe0");
    hls::stream<pkt128_t> s_pe1("s_pe1");
    hls::stream<pkt128_t> s_pe2("s_pe2");
    hls::stream<pkt128_t> s_pe3("s_pe3");

    hls::stream<pkt128_t> s_pkt0("s_pkt0");
    hls::stream<pkt128_t> s_pkt1("s_pkt1");
    hls::stream<pkt128_t> s_pkt2("s_pkt2");
    hls::stream<pkt128_t> s_pkt3("s_pkt3");

#pragma HLS STREAM variable=s_pe0  depth=512
#pragma HLS STREAM variable=s_pe1  depth=512
#pragma HLS STREAM variable=s_pe2  depth=512
#pragma HLS STREAM variable=s_pe3  depth=512
#pragma HLS STREAM variable=s_pkt0 depth=2048
#pragma HLS STREAM variable=s_pkt1 depth=2048
#pragma HLS STREAM variable=s_pkt2 depth=2048
#pragma HLS STREAM variable=s_pkt3 depth=2048

#pragma HLS DATAFLOW

    dispatch_route(a_packets, total_pkts, row_base, row_span,
                   s_pe0, s_pe1, s_pe2, s_pe3);

    pe_spmm_compute_topk(s_pe0, b_port0, row_base, nrows_act,
                         K, N, row_span, tn_size, num_cb, local_k, 0, b_sexp, s_pkt0);
    pe_spmm_compute_topk(s_pe1, b_port1, row_base, nrows_act,
                         K, N, row_span, tn_size, num_cb, local_k, 1, b_sexp, s_pkt1);
    pe_spmm_compute_topk(s_pe2, b_port2, row_base, nrows_act,
                         K, N, row_span, tn_size, num_cb, local_k, 2, b_sexp, s_pkt2);
    pe_spmm_compute_topk(s_pe3, b_port3, row_base, nrows_act,
                         K, N, row_span, tn_size, num_cb, local_k, 3, b_sexp, s_pkt3);

    collect_write(
        s_pkt0, s_pkt1, s_pkt2, s_pkt3,
        result_packets,
        row_base, nrows_act,
        row_span,
        local_k, top_k);
}
}