#!/bin/bash
#SBATCH --job-name=pt2x_500k
#SBATCH --output=pt2x_500k_%j.out
#SBATCH --error=pt2x_500k_%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=400G --time=16:00:00
#SBATCH --account torch_pr_554_general
# ==============================================================================
# EXACT PT2 (no eps_2 screening) at |S| = 5e5.  VALIDATION RUN.
# SUPERSEDES the version that died as job 15166440 (exit 137 = SIGKILL = OOM).
#
# ------------------------------------------------------------------------------
# WHAT WENT WRONG IN 15166440, AND THE FIX
#
# It ran PBATCH=5000, copied from the SCREENED 1e7 runs. That is safe when
# screening removes ~99% of emissions; it is not safe unscreened.
#
# From the one batch that completed: 5000 sources emitted 36,868,198,048 B of
# spill = 1.15e9 records, i.e. 2.30e5 raw contributions PER SOURCE (raw, before
# cross-batch dedup; the fully deduplicated figure at |S| = 2e5 was 7.32e4 per
# source). The screened 1e7 runs emitted 3.26e3 per source. So the unscreened
# accumulator holds ~71x more per batch:
#
#     PBATCH=5000, exact    -> 1.15e9 entries in RAM  ->  55-75 GB per drain
#     PBATCH=5000, screened -> 1.63e7 entries         ->  ~1 GB   (why 600G worked)
#
# Add 48 threads (P_in = 480 partition buffers, vs 160 at 16 threads) and the
# 200G request was not enough. Note the disk guard never fired -- peak was only
# 34.3 GiB against a 3200 GiB limit. This was purely a RAM failure.
#
# THREE CHANGES:
#   PBATCH  5000 -> 500   (the setting validated by the exact 2e5 run)
#   threads   48 -> 16    (matches that run; fewer partition buffers)
#   --mem   200G -> 400G  (headroom; expected use is well under 100G)
#
# CONFIRM THE DIAGNOSIS FIRST:
#     sacct -j 15166440 --format=JobID,State,ExitCode,MaxRSS,ReqMem
# If MaxRSS is at or near 200G, this is OOM and the settings below fix it.
# If MaxRSS is small, something else killed it -- do not resubmit blindly.
#
# ------------------------------------------------------------------------------
# PURPOSE OF THE RUN
# Second exact-vs-screened comparison, at 2.5x the size of the |S| = 2e5 one.
# Run on the SAME checkpoint as the screened 500k job, then compare
#     E_PT2(exact)  vs  E_PT2(eps2 = 2e-7)
# That difference is a DIRECT measurement of the screening bias in the regime
# where the benchmark series lives. Together with the 2e5 point it shows the
# bias falls with |S|, which is the argument for using eps2 = 2e-7 everywhere.
#
# E_var for this checkpoint = -293.1221680202 Ha (from 15166440's banner).
#
# ------------------------------------------------------------------------------
# SIZING (all from measured runs, not guesses)
#   raw per batch of 500      = 3.43 GiB
#   n_batches                 = 1000
#   final compacted           ~ 1091 GiB   (7.323e4 unique/source x 5e5 x 32 B)
#   COMPACT=50 -> interval raw 172 GiB -> peak ~ 1263 GiB
#   RAM per drain             ~ 6 GB
#   wall (16 thr, linear in sources from 5583 s at 2e5) ~ 3.9 h
# MAXDISK=2500 leaves ~1.9x margin over the projected peak and stays well under
# the 4193 GiB quota on a filesystem now at 98%.
# ==============================================================================
set -o pipefail
ulimit -n 65536 || ulimit -n 32768 || true
module purge
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
JB=/home/nh2429/julia/julia

SD=/scratch/nh2429/greedysparse/chemistry_experiments/scripts
BASE=/scratch/nh2429/greedysparse/chemistry_experiments/ozone_new_inputs/input_data/asci_gsno_om

# ===== EDIT =====
TAG=500k
CK=/scratch/nh2429/greedysparse/chemistry_experiments/ozone_new_inputs/results/checkpoints/asci__O3_asci_c25k_t500k.ckpt
# ================

POUT=0
PBATCH=500         # NOT 5000 -- see the note above
COMPACT=50         # 20 compactions over 1000 batches; holds peak near compacted
MAXDISK=2500       # GiB, Julia-side abort
HARDKILL_GIB=2700  # GiB, script-side SIGKILL
EPS=1e-3

HARDKILL_B=$(( HARDKILL_GIB * 1024 * 1024 * 1024 ))
[[ -f "$CK" ]] || { echo "no checkpoint: $CK"; exit 1; }
PSP=/scratch/nh2429/pt2x_${TAG}_${SLURM_JOB_ID}
PKF=$(mktemp)
cleanup() { rm -rf "$PSP" "$PKF"; }
trap cleanup EXIT
rm -rf "$PSP"; mkdir -p "$PSP"
echo "checkpoint: $CK   (EXACT PT2, no screening)"
echo "spill dir : $PSP"
echo "expect    : E_var = -293.1221680202 Ha"
df -h "$PSP"

t0=$(date +%s)
"$JB" -t "$JULIA_NUM_THREADS" "$SD/run_asci_pt2_ooc.jl" \
  90 24 3 "$EPS" "$BASE/eris.txt" "$BASE/hcore.txt" "$BASE/symm.txt" \
  "$CK" "$POUT" "$PBATCH" "$PSP" "$COMPACT" "$MAXDISK" \
  2> "$HOME/pt2x_${TAG}_${SLURM_JOB_ID}.err" &
JPID=$!

# Report RSS alongside disk: 15166440 died on memory while disk was trivial,
# so watching only disk would have missed it again.
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
      sleep 120
  done
) & MON=$!

wait "$JPID"; rc=$?
kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null
t1=$(date +%s)
PEAK_B=$(sort -n "$PKF" 2>/dev/null | tail -1); [[ -z "$PEAK_B" ]] && PEAK_B=0
awk -v p="$PEAK_B" -v w="$((t1-t0))" -v rc="$rc" \
    'BEGIN{printf "RESULT exact-500k  exit=%d  wall=%ds (%.2f h)  MEASURED PEAK = %.1f GiB\n",
           rc, w, w/3600, p/1073741824}'
echo "exit 137 = SIGKILL (OOM). exit 0 = success."
echo "COMPARE this E_PT2 against the eps2=2e-7 run on the same checkpoint."
exit $rc
