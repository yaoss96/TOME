#!/usr/bin/env python3
"""Add segmented-minimum Top-k maintenance to a generated variant.

ToME keeps each row's Top-k buffer unsorted with a running minimum: a
candidate at or below the minimum is rejected in one comparison, and an
accepted one re-derives the minimum with a k-entry scan.  HLS budgets for
every candidate being accepted, so the scan dominates its worst case.

This transformation splits the buffer into TK_SEG-entry segments with a
per-segment minimum.  Replacing one element dirties exactly one segment, so
an accept costs a TK_SEG-entry rescan plus a reduction over the segment
minima, both fully unrolled.  The result is exact: it changes how the
minimum is found, not which elements are kept (tb/tb_topk.cpp,
tb/tb_ties.cpp).

On the ZCU104 the segmented buffer bounds the worst case but, at an
initiation interval of 37 per candidate, ran 1.12-1.91x slower than the
full scan with identical output, because accepts are rare in practice.
ToME therefore keeps the scan; hls/tome_segmented_topk was generated with

    python gen_variants.py --out tome_segmented_topk --fastexp --sexp --bq
    python seg_topk.py tome_segmented_topk 16

Usage: seg_topk.py <variant_dir> [TK_SEG, default 8]
"""
import io
import os
import sys

d = sys.argv[1]
seg = int(sys.argv[2]) if len(sys.argv) > 2 else 8

h = io.open(os.path.join(d, 'spmm_accel.h'), encoding='utf-8').read()
c = io.open(os.path.join(d, 'spmm_accel.cpp'), encoding='utf-8').read()

# ---------------------------------------------------------------- header
anchor = '#define MAX_ROWS_PER_PE'
assert anchor in h, 'MAX_ROWS_PER_PE not found'
h = h.replace(anchor, """/* Segmented Top-k minimum: one dirty segment per replacement instead of
 * a full O(k) rescan.  TK_SEG is the unroll width of the segment rescan;
 * TK_NSEG segment minima are then reduced in a tree. */
#define TK_SEG   %d
#define TK_NSEG  ((MAX_LOCAL_K + TK_SEG - 1) / TK_SEG)

%s""" % (seg, anchor), 1)

# ---------------------------------------------------------------- insert
old_fn_start = 'static void topk_insert(float *tk_v, int *tk_c, int &tk_cnt,'
i0 = c.index(old_fn_start)
i1 = c.index('\n}\n', i0) + 3
new_fn = r"""static void topk_insert(float *tk_v, int *tk_c, int &tk_cnt,
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
"""
c = c[:i0] + new_fn + c[i1:]

# ---------------------------------------------------------------- drain
subs = [
    ("""    float tk_min_val[MAX_ROWS_PER_PE],
    int   tk_min_idx[MAX_ROWS_PER_PE],
    int   rs_pe,""",
     """    float tk_min_val[MAX_ROWS_PER_PE],
    int   tk_min_idx[MAX_ROWS_PER_PE],
    float seg_min[MAX_ROWS_PER_PE][TK_NSEG],
    int   seg_idx[MAX_ROWS_PER_PE][TK_NSEG],
    int   rs_pe,"""),
    ("""            topk_insert(tk_val[r], tk_col[r], tk_cnt[r],
                        tk_min_val[r], tk_min_idx[r],
                        lk, c_buf[r][j], cs + j);""",
     """            topk_insert(tk_val[r], tk_col[r], tk_cnt[r],
                        tk_min_val[r], tk_min_idx[r],
                        seg_min[r], seg_idx[r],
                        lk, c_buf[r][j], cs + j);"""),
    ("""                       tk_min_val, tk_min_idx,""",
     """                       tk_min_val, tk_min_idx,
                       tk_seg_min, tk_seg_idx,"""),
]
for old, new in subs:
    assert old in c, 'drain site missing: ' + old[:48].replace('\n', '|')
    c = c.replace(old, new, 1)

# ------------------------------------------------- declare + init arrays
old = """    float tk_min_val[MAX_ROWS_PER_PE];"""
assert old in c, 'tk_min_val declaration not found'
c = c.replace(old, """    float tk_seg_min[MAX_ROWS_PER_PE][TK_NSEG];
    int   tk_seg_idx[MAX_ROWS_PER_PE][TK_NSEG];
#pragma HLS ARRAY_PARTITION variable=tk_seg_min complete dim=2
#pragma HLS ARRAY_PARTITION variable=tk_seg_idx complete dim=2
    float tk_min_val[MAX_ROWS_PER_PE];""", 1)

old = """        tk_min_val[r] = 0.0f;"""
assert old in c, 'tk_min_val init not found'
c = c.replace(old, """        tk_min_val[r] = 0.0f;
        INIT_SEG: for (int s = 0; s < TK_NSEG; s++) {
#pragma HLS UNROLL
            tk_seg_min[r][s] = 3.4e38f;
            tk_seg_idx[r][s] = s * TK_SEG;
        }""", 1)

# the segment rescan reads TK_SEG consecutive entries in one cycle
old = '#pragma HLS BIND_STORAGE variable=tk_val type=RAM_2P impl=uram'
if old in c:
    c = c.replace(old, old +
                  '\n#pragma HLS ARRAY_PARTITION variable=tk_val cyclic factor=TK_SEG dim=2', 1)

io.open(os.path.join(d, 'spmm_accel.h'), 'w', encoding='utf-8', newline='\n').write(h)
io.open(os.path.join(d, 'spmm_accel.cpp'), 'w', encoding='utf-8', newline='\n').write(c)
print('segmented Top-k applied to %s (TK_SEG=%d)' % (d, seg))
