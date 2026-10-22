// Equivalence check: segmented-minimum Top-k vs the original O(k) rescan.
//
// The optimisation changes how the running minimum is located, not which
// elements survive, so the retained (value,column) multiset must match
// exactly on every trial -- not approximately.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "spmm_accel.h"

// ---- reference: the full-scan O(k) version ----
static void ref_insert(float *tk_v, int *tk_c, int &tk_cnt,
                       float &tk_min_v, int &tk_min_i,
                       int tk, float val, int col) {
    float aval = fabsf(val);
    if (aval < 1e-12f) return;
    if (tk_cnt < tk) {
        tk_v[tk_cnt] = val; tk_c[tk_cnt] = col;
        if (tk_cnt == 0 || aval < tk_min_v) { tk_min_v = aval; tk_min_i = tk_cnt; }
        tk_cnt++;
    } else {
        if (aval <= tk_min_v) return;
        tk_v[tk_min_i] = val; tk_c[tk_min_i] = col;
        float nm = fabsf(tk_v[0]); int ni = 0;
        for (int k = 1; k < tk; k++) {
            float ka = fabsf(tk_v[k]);
            if (ka < nm) { nm = ka; ni = k; }
        }
        tk_min_v = nm; tk_min_i = ni;
    }
}

// ---- the segmented version under test (mirrors the patched kernel) ----
static void seg_insert(float *tk_v, int *tk_c, int &tk_cnt,
                       float &tk_min_v, int &tk_min_i,
                       float *seg_min, int *seg_idx,
                       int tk, float val, int col) {
    float aval = fabsf(val);
    if (aval < 1e-12f) return;
    if (tk_cnt < tk) {
        int p = tk_cnt;
        tk_v[p] = val; tk_c[p] = col;
        int s = p / TK_SEG;
        if (aval < seg_min[s]) { seg_min[s] = aval; seg_idx[s] = p; }
        if (tk_cnt == 0 || aval < tk_min_v) { tk_min_v = aval; tk_min_i = p; }
        tk_cnt++;
    } else {
        if (aval <= tk_min_v) return;
        int p = tk_min_i;
        tk_v[p] = val; tk_c[p] = col;
        int s = p / TK_SEG, b = s * TK_SEG;
        float sm = 3.4e38f; int si = b;
        for (int t = 0; t < TK_SEG; t++) {
            int q = b + t;
            if (q < tk) { float qa = fabsf(tk_v[q]); if (qa < sm) { sm = qa; si = q; } }
        }
        seg_min[s] = sm; seg_idx[s] = si;
        float gm = 3.4e38f; int gi = 0;
        for (int t = 0; t < TK_NSEG; t++)
            if (t * TK_SEG < tk && seg_min[t] < gm) { gm = seg_min[t]; gi = seg_idx[t]; }
        tk_min_v = gm; tk_min_i = gi;
    }
}

int main() {
    srand(20261019);
    int bad = 0, trials = 0;
    long ref_ops = 0, seg_ops = 0;

    for (int k : {16, 50, 100, 200}) {
        for (int trial = 0; trial < 60; trial++) {
            int N = k * 8;
            std::vector<float> rv(k), sv(k);
            std::vector<int> rc(k), sc(k);
            std::vector<float> smin(TK_NSEG, 3.4e38f);
            std::vector<int> sidx(TK_NSEG, 0);
            int rcnt = 0, scnt = 0, rmi = 0, smi = 0;
            float rmv = 0, smv = 0;

            for (int i = 0; i < N; i++) {
                // heavy-tailed, like a propagated frontier
                float v = powf(10.0f, -4.0f * (float)rand() / RAND_MAX);
                int col = rand() % 100000;
                int before = rcnt;
                ref_insert(rv.data(), rc.data(), rcnt, rmv, rmi, k, v, col);
                seg_insert(sv.data(), sc.data(), scnt, smv, smi,
                           smin.data(), sidx.data(), k, v, col);
                if (before == k && rcnt == k) { ref_ops += k; seg_ops += TK_SEG + TK_NSEG; }
            }

            std::vector<std::pair<int,float>> a, b;
            for (int i = 0; i < rcnt; i++) a.push_back({rc[i], rv[i]});
            for (int i = 0; i < scnt; i++) b.push_back({sc[i], sv[i]});
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            trials++;
            if (a != b) { bad++; if (bad < 3) printf("  MISMATCH k=%d trial=%d\n", k, trial); }
        }
    }
    printf("retained-set equivalence: %d/%d trials identical\n", trials - bad, trials);
    printf("rescan work: reference %ld ops, segmented %ld ops  (%.1fx less)\n",
           ref_ops, seg_ops, ref_ops / (double)(seg_ops ? seg_ops : 1));
    return bad != 0;
}
