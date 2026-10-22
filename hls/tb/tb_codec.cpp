// Packet-level validation of the MP-COO codec, against the very header
// that gets synthesised.
//
// This mirrors greedy_pack_to_buf: group adjacent non-zeros, pick the tier
// from the group's dynamic range, place the group's maximum on the
// sub-format ceiling with the shared exponent, encode, decode, compare.
//
// Build:
//   g++ -std=c++14 -I<Vitis>/include -I<variant_dir> tb_codec.cpp
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "spmm_accel.h"

static double frand(double lo, double hi) {
    return lo + (hi - lo) * (double)rand() / (double)RAND_MAX;
}

struct Stat {
    long n = 0, flushed = 0;
    int tier_hist[3] = {0, 0, 0};
    std::vector<double> rel;
};

// Encode/decode one group exactly as the hardware packer does.
static void do_group(const std::vector<float>& v, Stat& st) {
    float mx = 0.0f, mn = 1e30f;
    for (size_t i = 0; i < v.size(); i++) {
        float a = fabsf(v[i]);
        if (a > mx) mx = a;
        if (a > 0.0f && a < mn) mn = a;
    }
    if (mx <= 0.0f) return;

#ifdef WITH_SEXP
    int prec = tier_for_range(mx, mn);
    int fmt  = subfmt_for_range(prec, mx, mn);
#else
    int prec = classify_prec(mx);
    int fmt  = (prec == PREC_FP8) ? fp8_select_pfmt(mx)
             : (prec == PREC_FP6) ? fp6_select_pfmt(mx) : 0;
#endif
    int eb, mb, bias;
    if (prec == PREC_FP8)      get_fp8_params(fmt, eb, mb, bias);
    else if (prec == PREC_FP6) get_fp6_params(fmt, eb, mb, bias);
    else                       get_fp4_params(eb, mb, bias);
    st.tier_hist[prec]++;

#ifdef WITH_SEXP
    const int sh = fexp_of(mx) - fmax_exp(prec, fmt);
#else
    const int sh = 0;
#endif

    for (size_t i = 0; i < v.size(); i++) {
#ifdef WITH_SEXP
        ap_uint<8> e = mfp_encode(fscale2(v[i], -sh), eb, mb, bias);
        float back   = fscale2(mfp_decode(e, eb, mb, bias), sh);
#else
        ap_uint<8> e = mfp_encode(v[i], eb, mb, bias);
        float back   = mfp_decode(e, eb, mb, bias);
#endif
        st.n++;
        if (v[i] != 0.0f && back == 0.0f) { st.flushed++; continue; }
        st.rel.push_back(fabs((double)back - (double)v[i]) / fabs((double)v[i]));
    }
}

int main() {
    srand(20261007);
    // Decades a contracting row-stochastic frontier actually visits, and
    // the within-packet spread the greedy packer actually sees (adjacent
    // columns have similar magnitudes, so the spread is modest).
    const double decades[] = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    const double spreads[] = {2.0, 8.0, 64.0};

    printf("%-9s %-8s %9s %10s %12s %12s   %s\n", "decade", "spread",
           "flush0%", "med_rel", "p99_rel", "max_rel", "tiers FP8/FP6/FP4");
    printf("%s\n", "-------------------------------------------------"
                   "--------------------------------------");
    for (int d = 0; d < 6; d++) {
        for (int s = 0; s < 3; s++) {
            Stat st;
            for (int g = 0; g < 20000; g++) {
                int cnt = 1 + rand() % 7;
                double base = decades[d];
                std::vector<float> v(cnt);
                for (int i = 0; i < cnt; i++)
                    v[i] = (float)frand(base / spreads[s], base);
                do_group(v, st);
            }
            std::sort(st.rel.begin(), st.rel.end());
            double med = st.rel.empty() ? -1 : st.rel[st.rel.size() / 2];
            double p99 = st.rel.empty() ? -1 : st.rel[(size_t)(st.rel.size() * 0.99)];
            double mxr = st.rel.empty() ? -1 : st.rel.back();
            printf("%-9.0e %-8.0f %8.2f%% %10.4f %12.4f %12.4f   %d/%d/%d\n",
                   decades[d], spreads[s],
                   100.0 * st.flushed / (st.n ? st.n : 1), med, p99, mxr,
                   st.tier_hist[PREC_FP8], st.tier_hist[PREC_FP6],
                   st.tier_hist[PREC_FP4]);
        }
    }
    return 0;
}
