#!/bin/sh
# Build a bitstream for the ZCU104: HLS C synthesis, IP export, Vivado.
#
#   scripts/build_zcu104.sh [variant] [pl_mhz]
#
# variant is a directory under hls/ (default: tome).  The PL clock defaults
# to 150 MHz for tome and 100 MHz otherwise; the HLS clock target is set to
# match.  Vitis HLS and Vivado 2024.2 must be on PATH, or named by the
# VITIS_HLS and VIVADO variables.  Output: build/<variant>/out/<variant>_zcu104.{bit,hwh}
set -eu
VARIANT=${1:-tome}
if [ "$VARIANT" = tome ]; then DEF_MHZ=150; else DEF_MHZ=100; fi
MHZ=${2:-$DEF_MHZ}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HLS=${VITIS_HLS:-vitis_hls}
VIV=${VIVADO:-vivado}
SRC="$ROOT/hls/$VARIANT"
BUILD="$ROOT/build/$VARIANT"

[ -f "$SRC/spmm_accel.cpp" ] || { echo "no such variant: $SRC"; exit 1; }
mkdir -p "$BUILD"
cp "$SRC/spmm_accel.cpp" "$SRC/spmm_accel.h" "$SRC/run.tcl" "$ROOT/hls/export.tcl" "$BUILD/"
PERIOD=$(awk "BEGIN { printf \"%.3f\", 1000 / $MHZ }")
sed "s/^create_clock -period [0-9.]*/create_clock -period $PERIOD/" "$SRC/run.tcl" > "$BUILD/run.tcl"
cd "$BUILD"

echo "== HLS C synthesis, $VARIANT at $MHZ MHz ($(date +%H:%M:%S))"
"$HLS" -f run.tcl > csynth.log 2>&1 || { tail -20 csynth.log; exit 1; }
grep -E "Estimated Fmax" csynth.log || true

echo "== IP export ($(date +%H:%M:%S))"
"$HLS" -f export.tcl > export.log 2>&1 || { tail -20 export.log; exit 1; }
[ -f cmp/hls/impl/ip/component.xml ] || { echo "IP export failed, see export.log"; exit 1; }

echo "== Vivado ($(date +%H:%M:%S))"
"$VIV" -mode batch -nojournal -log vivado.log -source "$ROOT/vivado/build_zcu104.tcl" \
    -tclargs "$BUILD/vivado" "$BUILD/cmp/hls/impl/ip" "$BUILD/out" "${VARIANT}_zcu104" "$MHZ" \
    > vivado.out 2>&1 || { tail -30 vivado.out; exit 1; }
grep -A3 "WNS(ns)" "$BUILD/vivado/${VARIANT}_zcu104.runs/impl_1/design_1_wrapper_timing_summary_routed.rpt" 2>/dev/null | sed -n 3p || true
ls -l "$BUILD/out"
echo "== done ($(date +%H:%M:%S))"
