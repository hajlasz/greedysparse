#!/bin/bash
#SBATCH --job-name=pt2s
#SBATCH --output=pt2s_%j.out
#SBATCH --error=pt2s_%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48
#SBATCH --mem=300G --time=08:00:00
#SBATCH --account torch_pr_554_general
# ==============================================================================
# SCREENED PT2 (eps2 = 2e-7) for the benchmark series.
# Uses run_asci_pt2_ooc_eps2.jl -- NOT run_asci_pt2_ooc.jl.
#
# ONE FILE, SIX RUNS: edit TAG / CK / and the two SBATCH lines per the table:
#
#   |S|     TAG    peak GiB   wall    --time      --mem
#   5e5     500k       230    0.7 h   04:00:00     300G
#   5e5     500kB      230    0.7 h   04:00:00     300G   (N_cdets=1e5 variant)
#   1e6     1M         326    0.9 h   04:00:00     300G
#   2e6     2M         461    1.3 h   04:00:00     300G
#   3e6     3M         564    1.6 h   06:00:00     300G
#   5e6     5M         729    2.1 h   08:00:00     300G
#
# WHY eps2 = 2e-7 EVERYWHERE. The screening bias at fixed eps2 SHRINKS with |S|,
# because the eigenvector is normalized: typical |x_i| ~ |S|^(-1/2), so a fixed
# eps2 cuts a smaller fraction of the correction at small |S|. At |S| = 1e7 the
# bias at 2e-7 is bounded at <= 0.18 mHa (7-point ladder, job 14992593); every
# point below 1e7 is therefore biased by LESS. Holding eps2 fixed also means the
# residual bias is common-mode across the series, so it largely cancels in the
# slope b and shifts the intercept far less than its absolute size suggests.
#
# ESTIMATES are anchored on job 14992593 (|S| = 1e7, eps2 = 2e-7, 48 threads:
# 1030.2 GiB peak, 2.98 h) scaled as externals/source ~ |S|^(-1/2). Requests
# carry ~3x headroom on wall time.
# ==============================================================================
set -o pipefail
ulimit -n 65536 || ulimit -n 32768 || true
module purge
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
JB=/home/nh2429/julia/julia

SD=/scratch/nh2429/greedysparse/chemistry_experiments/scripts
BASE=/scratch/nh2429/greedysparse/chemistry_experiments/ozone_new_inputs/input_data/gsno_om

# ===== EDIT (3 lines) =====
TAG=500k
CK=/scratch/nh2429/greedysparse/chemistry_experiments/ozone_new_inputs/results/checkpoints/asci__O3_asci_c25k_t500k.ckpt
EXPECT_EVAR="check against the ASCI job's banner"
# ==========================

POUT=0
PBATCH=5000        # matches the validated 1e7 runs
COMPACT=50         # matches the validated 1e7 runs; the FIRST interval sets the peak
MAXDISK=3500       # GiB, Julia-side abort
HARDKILL_GIB=3600  # GiB, script-side SIGKILL between compactions
EPS=1e-3
EPS2=2e-7

HARDKILL_B=$(( HARDKILL_GIB * 1024 * 1024 * 1024 ))
[[ -f "$CK" ]] || { echo "no checkpoint: $CK"; exit 1; }
PSP=/scratch/nh2429/pt2s_${TAG}_${SLURM_JOB_ID}
PKF=$(mktemp)
cleanup() { rm -rf "$PSP" "$PKF"; }
trap cleanup EXIT
rm -rf "$PSP"; mkdir -p "$PSP"
echo "checkpoint: $CK"
echo "spill dir : $PSP    eps2 = $EPS2"
echo "expect    : $EXPECT_EVAR"
df -h "$PSP"

t0=$(date +%s)
"$JB" -t "$JULIA_NUM_THREADS" "$SD/run_asci_pt2_ooc_eps2.jl" \
  90 24 3 "$EPS" "$BASE/eris.txt" "$BASE/hcore.txt" "$BASE/symm.txt" \
  "$CK" "$POUT" "$PBATCH" "$PSP" "$COMPACT" "$MAXDISK" "$EPS2" \
  2> "$HOME/pt2s_${TAG}_${SLURM_JOB_ID}.err" &
JPID=$!

# The Julia MAXDISK check only fires at compaction points, so sample here too.
(
  while kill -0 "$JPID" 2>/dev/null; do
      B=$(du -sb "$PSP" 2>/dev/null | cut -f1)
      if [[ -n "$B" ]]; then
          echo "$B" >> "$PKF"
          awk -v b="$B" 'BEGIN{printf "SPILL %d B = %.2f GiB\n", b, b/1073741824}'
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
exit $rc
