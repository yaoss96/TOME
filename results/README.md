# Measured data

Everything the paper reports, as measured. The scripts in `sim/` and
`analysis/` turn these files into the tables, figures and numbers of the
paper.

| Paper | Files | Script |
|---|---|---|
| Table I, B0 time in Sec. IV-C | `zcu104/board_100mhz/b0_*.json` | none |
| Table II, Fig. 3 | `fidelity/sweep_results.json` | `sim/tables.py`, `analysis/fig3_fidelity.py` |
| Table III, Sec. IV-C | `hls/*_csynth.rpt`, `hls/tome_150mhz_*.rpt` | none |
| Sec. IV-C | `zcu104/board_100mhz/` | `analysis/numbers.py` |
| Fig. 4, Sec. IV-D | `zcu104/time/`, `zcu104/power/` | `analysis/fig4_compare.py`, `analysis/numbers.py` |

## fidelity/

`sweep_results.json` holds one record per graph, hop count (2, 3, 5), k
(50, 100, 200) and codec, written by `sim/sweep.py` with M = 16 query rows.

| Field | Meaning |
|---|---|
| `overlap`, `mass_recall` | the last hop scored against the exact FP64 chain |
| `bits_per_nz` | packet bits per non-zero, headers and empty slots included |
| `zero_pct` | share of retained values that decode to zero |
| `partial_pct` | share of packets not filled |
| `tiers` | packet count per precision tier |

| `cfg` | Codec |
|---|---|
| `fp8` | single-tier FP8 |
| `two` | magnitude-tiered FP6/FP8 |
| `mixed` | magnitude-tiered FP4/FP6/FP8, MT in the paper |
| `mixed_sexp` | MT with a shared exponent |
| `mixed_dyn` | ToME |
| `fp32_exact` | unquantized check |

## hls/

| File | What it is |
|---|---|
| `*_csynth.rpt` | Vitis HLS 2024.2 synthesis summaries (module table) of the designs in `hls/` |
| `tome_100mhz_csynth.rpt` | ToME at a 10-ns target, as in Table III |
| `tome_150mhz_csynth.rpt` | ToME at the 6.667-ns target of the deployed bitstream |
| `tome_150mhz_utilization_placed.rpt`, `tome_150mhz_timing_summary_routed.rpt` | Vivado reports of that bitstream |

## zcu104/

All runs are on the same instance. S is a dense row-wise softmax over each
graph with a constant background logit along each row. The instance has
M = 256 query rows, each starting with 1–4 active states. The field
`inst` fingerprints the instance (`host/tome/problem.py`). Times are in ms.

### time/

Fig. 4: three hops at k = 100 on six graphs, FPGA designs at 150 MHz.
Every hop is fed the exact FP64 frontier of the previous hop, so every
engine does identical work.

| File | Engine | Time field |
|---|---|---|
| `tome_<graph>.json` | ToME | `kernel_ms` per hop and in total |
| `a53_<graph>.json` | `sparse_dot_topn` on the four Cortex-A53 cores, FP32 | `ms` per hop |
| `tcad_<graph>.json` | Li et al.'s SpGEMM accelerator, writing C to DDR, with the A53 selecting Top-k | `total_ms` per hop, the whole pipeline |
| `dac21_<graph>.json` | Parravicini et al.'s Top-k SpMV engine (Q8.12), one launch per frontier row | `kernel_ms` |
| `dytopk_<graph>.json` | DyTopK (FP4), one launch per frontier row | `kernel_ms` |
| `gbench2_<engine>_<graph>.json` | the A53 gathering and packing S's rows for every launch of an SpMV engine | `total_pack_ms` |
| `*_rerun.json` | a repeated run | per hop, the minimum of the two runs is used |

The fidelity fields of every engine file:

| Field | Meaning |
|---|---|
| `overlap_tie` | tie-aware Top-k overlap with the exact chain, the score the paper reports |
| `mass_tie` | retained-mass recall, tie-aware |
| `overlap`, `mass_recall` | plain scores, which count tie-breaks as misses |
| `out_sha` | ToME only: hash of each hop's output, identical across repetitions |

### power/

Fig. 4(b) is the 12-V board input, read from the board's INA226 every
10 ms.
- **Method.** Each design idles for 2 s and then repeats its timed work back
  to back for about 6 s.
- **Fields.** `idle_w`, `run_w` and `dyn_w = run_w - idle_w`.
- **A53.** It was measured with the PL clock stopped.
- **SpMV engines.** `kernel` and `host` are measured separately: the kernel
  relaunching prepacked rows, and the A53 packing.

### board_100mhz/

The 100-MHz runs of Sec. IV-C and Table I. Times are per hop, the minimum
over runs.

- **`summary.json`**, entries keyed `engine|bitstream|graph|N|k`, covers:
  - ToME (`tome_zcu.bit`) and its segmented Top-k variant
    (`tome_zcu_ftk.bit`);
  - the A53 with fused (`a53_sdt`) and unfused (`a53_scipy`) Top-k.
- **`b0_email_k200.json`, `b0_erdos972_k200.json`** hold B0, k = 200.
  Per hop they give the kernel time and the DRAM bytes it moved, S
  (`B_MB`) and the frontier (`A_KB`, `C_KB`).
