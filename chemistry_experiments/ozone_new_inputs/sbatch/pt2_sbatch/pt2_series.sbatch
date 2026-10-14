#!/bin/bash
#SBATCH --job-name=O3_pt2_series
#SBATCH --output=O3_pt2_%j.out
#SBATCH --error=O3_pt2_%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48
#SBATCH --mem=300G --time=08:00:00
#SBATCH --account=torch_pr_554_general
# ==============================================================================
# SCREENED PT2 (eps2 = 2e-7) for the benchmark series.  ONE FILE, THREE RUNS.
#
# Takes the SAME N_TDETS / N_CDETS as asci_series.sbatch and derives the
# checkpoint path with the identical TAG rule, so the path is never typed by
# hand and can never point at the wrong run:
#
#   sbatch --export=ALL,N_TDETS=1000000,N_CDETS=50000  pt2_series.sbatch
#   sbatch --export=ALL,N_TDETS=2000000,N_CDETS=100000 pt2_series.sbatch
#   sbatch --export=ALL,N_TDETS=3000000,N_CDETS=150000 pt2_series.sbatch
#
# Uses run_asci_pt2_ooc_eps2.jl (14 arguments) -- NOT run_asci_pt2_ooc.jl.
#
# WHY eps2 = 2e-7 ON EVERY POINT. The HCI/ASCI extrapolation convention holds
# the perturbative screening parameter FIXED across the series and varies only
# the variational size; that keeps the residual screening bias common-mode, so
# it largely cancels in the slope. 2e-7 is the value used for the 1e7 point,
# where a 7-point ladder bounded its bias at <= 0.18 mHa. The bias at fixed
# eps2 SHRINKS as |S| falls (the eigenvector is normalised, so |x_i| ~
# |S|^(-1/2) and a fixed cut removes a smaller fraction), so every point below
# 1e7 is biased by less than that.
#
# ESTIMATES (anchored on job 14992593: |S| = 1e7, eps2 = 2e-7, 48 threads,
# 1030.2 GiB peak, 2.98 h; scaled as externals/source ~ |S|^(-1/2)):
#
#   N_TDETS   peak GiB   wall     expected |E_PT2|
#     1e6        326      0.9 h       45.2 mHa
#     2e6        461      1.3 h       37.1 mHa
#     3e6        564      1.6 h       33.1 mHa
#
# Run these ONE AT A TIME. /scratch is at 98% and shared.
#
# TWO DISK GUARDS: MAXDISK inside Julia (checked only at compaction points) and
# a du monitor here that SIGKILLs between them. The second exists because the
# first has blind spots -- that is how the true peak was under-reported before.
# ==============================================================================
set -o pipefail
ulimit -n 65536 || ulimit -n 32768 || true
module purge
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
JB=/home/nh2429/julia/julia

BASE_DIR=/scratch/nh2429/greedysparse
SD=${BASE_DIR}/chemistry_experiments/scripts
BASE=${BASE_DIR}/chemistry_experiments/ozone_new_inputs/input_data/asci_gsno_om

N_CDETS=${N_CDETS:-50000}
N_TDETS=${N_TDETS:-1000000}

POUT=0
PBATCH=5000        # validated at |S| = 1e7 WITH screening.  Do NOT reuse this
                   # for an unscreened run: exact PT2 emits ~71x more per batch
                   # and 5000 will OOM (this is what killed job 15166440).
COMPACT=50
MAXDISK=3500       # GiB, Julia-side abort
HARDKILL_GIB=3600  # GiB, script-side SIGKILL
EPS=1e-3
EPS2=2e-7

# Same TAG rule as asci_series.sbatch -- keep the two in sync.
human() {
    local n=$1
    if   [ $((n % 1000000)) -eq 0 ]; then echo "$((n / 1000000))M"
    elif [ $((n % 1000))    -eq 0 ]; then echo "$((n / 1000))k"
    else echo "${n}"; fi
}
TAG="O3_asci_c$(human ${N_CDETS})_t$(human ${N_TDETS})"
CK=${BASE_DIR}/chemistry_experiments/ozone_new_inputs/results/checkpoints/asci__${TAG}.ckpt

HARDKILL_B=$(( HARDKILL_GIB * 1024 * 1024 * 1024 ))
[[ -f "$CK" ]] || { echo "no checkpoint: $CK"; echo "run asci_series.sbatch first"; exit 1; }

PSP=/scratch/nh2429/pt2s_${TAG}_${SLURM_JOB_ID}
PKF=$(mktemp)
cleanup() { rm -rf "$PSP" "$PKF"; }
trap cleanup EXIT
rm -rf "$PSP"; mkdir -p "$PSP"
echo "=============================================================="
echo " tag        : ${TAG}"
echo " checkpoint : ${CK}"
echo " spill dir  : ${PSP}"
echo " eps2       : ${EPS2}   (same value on every point in the series)"
echo "=============================================================="
df -h "$PSP"

t0=$(date +%s)
"$JB" -t "$JULIA_NUM_THREADS" "$SD/run_asci_pt2_ooc_eps2.jl" \
  90 24 3 "$EPS" "$BASE/eris.txt" "$BASE/hcore.txt" "$BASE/symm.txt" \
  "$CK" "$POUT" "$PBATCH" "$PSP" "$COMPACT" "$MAXDISK" "$EPS2" \
  2> "$HOME/pt2s_${TAG}_${SLURM_JOB_ID}.err" &
JPID=$!

# Report RSS as well as disk: job 15166440 died on memory while disk was
# trivial, so a disk-only monitor would miss that failure mode again.
(
  while kill -0 "$JPID" 2>/dev/null; do
      B=$(du -sb "$PSP" 2>/dev/null | cut -f1)
      R=$(ps -o rss= -p "$JPID" 2>/dev/null | tr -d ' ')
      if [[ -n "$B" ]]; then
          echo "$B" >> "$PKF"
          awk -v b="$B" -v r="${R:-0}" \
              'BEGIN{printf "SPILL %.2f GiB   RSS %.1f GB\n", b/1073741824, r/1048576}'
          if [[ "$B" -gt "$HARDKILL_B" ]]; then
              echo "*** HARD KILL: exceeds ${HARDKILL_GIB} GiB ***"
              kill -9 "$JPID" 2>/dev/null; break
          fi
      fi
      sleep 60
  done
) & MON=$!

wait "$JPID"; rc=$?
kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null
t1=$(date +%s)
PEAK_B=$(sort -n "$PKF" 2>/dev/null | tail -1); [[ -z "$PEAK_B" ]] && PEAK_B=0
awk -v p="$PEAK_B" -v w="$((t1-t0))" -v rc="$rc" -v t="$TAG" \
    'BEGIN{printf "RESULT %s  exit=%d  wall=%ds (%.2f h)  MEASURED PEAK = %.1f GiB\n",
           t, rc, w, w/3600, p/1073741824}'
echo "RECORD for the regression:  |S| = ${N_TDETS},  E_var,  E_PT2,  y = E_var + E_PT2"
exit $rc
