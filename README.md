# ToME: Shared-Exponent Mixed Precision for Iterative Top-k SpMM on Edge FPGAs

Multi-hop knowledge-graph reasoning repeatedly applies Top-k sparse–dense
matrix multiplication, P ← TopK(P·S), using each pruned output as the next
hop's sparse input. ToME is an accelerator for edge SoC FPGAs that fuses
Top-k selection with a row-wise (Gustavson) SpMM dataflow and keeps every
hop in one mixed-precision sparse format, **MP-COO**, so the embedded CPU
does no work between hops.

- **MP-COO** packs a row's non-zeros into 128-bit packets of five FP8, six
  FP6 or seven FP4 values. A per-packet shared exponent is funded by
  narrowing the row index to a panel-local one, and the precision tier is
  chosen by the packet's dynamic range. On a frontier that contracts at every
  hop, this raises Top-k overlap with an exact FP64 run from 0.33
  (magnitude-tiered FP8/FP6/FP4) to 0.97, while moving 14.8% fewer bits than
  FP8.
- **The kernel is compute-bound** even behind the ZCU104's one shared DDR
  port, so ToME stores the dense operand S in FP8 E4M3. A 128-bit beat then
  feeds 16 MAC lanes per PE instead of 4.
- **Results.** On a ZCU104 with every FPGA design at 150 MHz, ToME is
  6.7–12.6× faster and 6.3–11.9× more energy-efficient than the on-chip
  quad-core Cortex-A53. It is 1.5–91× faster than an SpGEMM accelerator and
  two Top-k SpMV accelerators, and 24–51× faster than the SpMV engines once
  the host preparation they need for every launch is counted.

This repository holds:
- the HLS source of every design in the paper;
- the Vivado flow for the ZCU104;
- the PYNQ host runtime;
- the software codec model and fidelity sweep;
- the measured data behind every table and figure, with the scripts that
  turn that data into them.

## Layout

```
hls/        HLS C++ of the four ZCU104 designs, the generator that derives
            ToME from the baseline B0, and unit testbenches
vivado/     block design and bitstream Tcl for the ZCU104
scripts/    build_zcu104.sh: HLS C synthesis -> IP export -> bitstream
host/       PYNQ runtime: MP-COO packer, problem instance, kernel driver,
            run_tome.py
sim/        software model of the codecs and the fidelity sweep
            (Table II, Fig. 3)
analysis/   Fig. 3, Fig. 4 and the measured numbers of Sec. IV, from results/
results/    measured data behind the paper (see results/README.md)
data/       get_graphs.py: the ten SuiteSparse graphs
```

## Designs

Every design has 4 PEs, k ≤ 200 and T_N = 256 on the ZU7EV
(xczu7ev-ffvc1156-2-e).

| Directory | Design (Table III) | LUT | DSP | BRAM (18 Kb) |
|---|---|---|---|---|
| `hls/b0` | B0: FP32 S, 4 lanes, `powf` codec, magnitude tiers | 308.1k | 804 | 312 |
| `hls/b0_powfree_sexp` | + pow-free codec, shared exponent | 156.6k | 208 | 252 |
| `hls/tome` | ToME: + dynamic tiers, FP8 S, 16 lanes | 204.6k | 448 | 492 |
| `hls/tome_segmented_topk` | ToME with a segmented Top-k buffer (Sec. III-C) | 248.7k | 456 | 596 |

These are HLS estimates at 100 MHz (`results/hls`). Placed and routed at
150 MHz, ToME uses:
- 68.5% of the LUTs;
- 75.0% of the BRAM;
- 25.5% of the DSPs;
- all 96 URAMs.

It closes timing with +0.24 ns of slack.

`hls/tome` and `hls/tome_segmented_topk` are generated from `hls/b0`. Each
of the paper's design decisions is a source transformation in
`hls/gen_variants.py`:

```
python hls/gen_variants.py --out hls/tome --fastexp --sexp --bq --period 6.667

python hls/gen_variants.py --out hls/tome_segmented_topk --fastexp --sexp --bq
python hls/seg_topk.py hls/tome_segmented_topk 16
```

`hls/b0_powfree_sexp` is kept as the design point synthesized for row 2 of
Table III. It predates two fixes that `gen_variants.py --sexp` contains:
the shared exponent places a packet's maximum at the format's ceiling, and
the dispatcher reads the 8-bit panel-local row. Without them it is not
functionally correct, so use it for its resource estimate only.

### Unit testbenches

The testbenches build with any C++14 compiler against the Vitis HLS
headers (`<Vitis>/include` holds `ap_int.h`):

```
g++ -std=c++14 -O2 -DWITH_SEXP -I<Vitis>/include -Ihls/tome hls/tb/tb_codec.cpp -o tb_codec
g++ -std=c++14 -O2 -I<Vitis>/include -Ihls/tome_segmented_topk hls/tb/tb_topk.cpp -o tb_topk
g++ -std=c++14 -O2 -I<Vitis>/include -Ihls/tome_segmented_topk hls/tb/tb_ties.cpp -o tb_ties
```

- `tb_codec` round-trips grouped values through the synthesized minifloat
  codec. Without `-DWITH_SEXP`, against `hls/b0`, it shows how magnitude
  tiers flush small values.
- `tb_topk` and `tb_ties` check that the segmented Top-k buffer keeps
  exactly the elements the full scan keeps, including on tie-heavy
  quantized data.

## Building a bitstream

The flow needs Vitis HLS and Vivado 2024.2:

```
VITIS_HLS=<path>/vitis_hls VIVADO=<path>/vivado scripts/build_zcu104.sh tome
```

- The output is `build/tome/out/tome_zcu104.bit` with its `.hwh`.
- PL0 is 150 MHz from IOPLL.
- The kernel's four AXI masters share one 128-bit HP port through a
  SmartConnect.
- The build took about 1.5 hours on our workstation, most of it in IP export
  and Vivado implementation.

## Running on the ZCU104

The runtime runs on the ZCU104's PYNQ 3.0 image. It needs NumPy and SciPy,
which the PYNQ virtual environment provides.

1. Copy `host/` and the `.bit`/`.hwh` pair to the board.
2. Run from a directory other than `/home/xilinx`. The image's
   `/home/xilinx/pynq` source tree shadows the installed package there.
3. Run as root:

```
sudo env -u PYTHONPATH XILINX_XRT=/usr BOARD=ZCU104 /usr/local/share/pynq-venv/bin/python3 \
    run_tome.py email.mtx --bit tome_zcu104.bit --mhz 150 --k 100 --hops 3 \
    --force --out tome_email.json
```

`run_tome.py`:
- builds the problem instance of the paper from the graph: a dense row-wise
  softmax S and 256 query rows;
- sets the PL clock and reads it back from the clock registers;
- runs three hops of P ← TopK(P·S).

For every hop it prints:
- the kernel time;
- a hash of the output;
- the overlap with the exact FP64 chain, both plain and tie-aware.

`--force` feeds every hop the exact frontier of the previous one, as in
Fig. 4. For email it reproduces `results/zcu104/time/tome_email.json`:
- kernel times of 7.82, 26.85 and 29.60 ms;
- tie-aware overlaps of 0.960, 0.989 and 0.982;
- the same output hashes.

## Reproducing the numbers without hardware

```
pip install -r requirements.txt
python sim/tables.py                          # Table II
python analysis/fig3_fidelity.py fig3.pdf     # Fig. 3
python analysis/fig4_compare.py fig4.pdf      # Fig. 4
python analysis/numbers.py                    # measured numbers of Sec. IV-C/IV-D
```

The fidelity sweep takes under a minute:

```
python data/get_graphs.py
python sim/sweep.py --out sweep.json
python sim/tables.py sweep.json
```

## Notes

- **Baselines.** Fig. 4 compares ToME with reproductions of three designs,
  run on the same board:
  - Li et al.'s Gustavson SpGEMM accelerator, with the A53 selecting Top-k;
  - the Top-k SpMV engine of Parravicini et al.;
  - DyTopK.

  Their measured results are in `results/zcu104`; their code is not part of
  this repository.
- **Graphs.** The SuiteSparse graphs are not redistributed.
  `data/get_graphs.py` downloads them from the SuiteSparse Matrix
  Collection.

## Authors

Shangshang Yao (Information Engineering University) and Zuoning Chen
(National Research Centre of Parallel Computer Engineering and Technology).
The paper is under review; a citation will be added once it is published.

## License

MIT; see [LICENSE](LICENSE).
