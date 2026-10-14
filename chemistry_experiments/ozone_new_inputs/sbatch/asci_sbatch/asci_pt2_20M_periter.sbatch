#!/bin/bash
#SBATCH --job-name=O3_20M_asci_pt2
#SBATCH --output=O3_20M_asci_pt2_%j.out
#SBATCH --error=O3_20M_asci_pt2_%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=600G
#SBATCH --account=torch_pr_554_general
# ==============================================================================
# Ozone |S| = 2e7, N_cdets = 1e6: ASCI advanced ONE iteration at a time, with a
# screened PT2 (eps2 = 1e-7) evaluated on the checkpoint after each iteration.
#
# Output is a per-iteration trajectory of BOTH E_var and E_var + E_PT2, i.e.
# the data for an "ASCI+PT2 vs iteration" convergence curve rather than a
# single horizontal line.
#
# ------------------------------------------------------------------------------
# HOW THE ONE-ITERATION-AT-A-TIME LOOP WORKS
#
# run_asci.jl takes Q_MAX as a cap on the ITERATION NUMBER, not a count per
# invocation (job 15526653 resumed at iteration 3 with Q_MAX = 20 and reported
# "Iterations: 5 (run from 4)").  So invocation k is launched with Q_MAX = k
# and RESUME = true, which advances the checkpoint by exactly one iteration.
#
# The checkpoint is OVERWRITTEN by each iteration, so it is archived to
# ..._iter${k}.ckpt before PT2 runs.  That archive is also what makes this
# script idempotent: on a resubmit, any iteration whose archive and PT2 log
# already exist is skipped.  Re-submitting the same script after a timeout
# simply continues.
#
# ------------------------------------------------------------------------------
# COST (measured; ASCI from jobs 15263821/15474431/15526653, PT2 projected from
# the 2e7 ladder job 15553879 with D ~ eps2^-1.524, T ~ eps2^-1.254)
#
#   iter   ASCI          PT2 @ 1e-7        running total
#   1      0.01 h        negligible        0.0 h     (|S| = 91k after iter 1)
#   2      4.51 h        8.6 h            13.1 h
#   3      9.03 h        8.6 h            30.7 h
#   4     10.48 h        8.6 h            49.8 h
#   5      9.37 h        8.6 h            67.8 h
#
# ~68 h against a 96 h limit.  Slack is real but not generous; the script is
# restartable precisely because that margin may not hold.
#
# ------------------------------------------------------------------------------
# DISK -- THE MAIN RISK.  READ THIS BEFORE SUBMITTING.
#
# PT2 at eps2 = 1e-7 on the CONVERGED |S| = 2e7 space projects to 3349 GiB,
# which is 98% of MAXDISK = 3400 and 80% of the 4193 GiB (4.5 TB) quota.  There
# is almost no margin, and EARLIER iterations are worse: the iteration-2 space
# had 1663 nnz/row against 1408 at iteration 5, so its PT2 could need ~3950 GiB
# and will abort.
#
# That is expected and handled: a failed PT2 is logged and the loop CONTINUES to
# the next ASCI iteration.  Do not treat a PT2 failure at iteration 2 or 3 as a
# job failure.  If you would rather not spend ~9 h discovering that, set
# PT2_FROM_ITER=4 below and only the well-converged spaces get a PT2.
#
# Nothing else of yours should touch /scratch while this runs.
# ==============================================================================

set -o pipefail
module purge
ulimit -n 65536 || ulimit -n 32768 || true
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}

JB=/home/nh2429/julia/julia
BASE_DIR=/scratch/nh2429/greedysparse
SD=${BASE_DIR}/chemistry_experiments/scripts

# ASCI reads integrals from greedy_sparse_clean (this is what produced every
# existing checkpoint); PT2 reads the copy under greedysparse.  E_var has
# matched exactly across the two at 2e5, 5e5, 1e7 and 2e7, so they are the same
# data -- do NOT "tidy" them into one root.
O3_DIR=/scratch/nh2429/greedy_sparse_clean/chemistry_experiments/ozone_new_inputs/input_data/asci_gsno_om
PT2_DIR=${BASE_DIR}/chemistry_experiments/ozone_new_inputs/input_data/asci_gsno_om

ERI=${O3_DIR}/eris.txt;  HCORE=${O3_DIR}/hcore.txt;  SYMM=${O3_DIR}/symm.txt
for f in "$ERI" "$HCORE" "$SYMM" "$PT2_DIR/eris.txt"; do
    [[ -s "$f" ]] || { echo "MISSING OR EMPTY INPUT: $f"; exit 1; }
done

# ============================ CONFIG ==========================================
NORB_TOT=90; NELEC_TOT=24; NORB_FRZN=3; EPSILON=1e-3

N_CDETS=1000000
N_TDETS=20000000
MAX_ITER=5              # iterations to attempt (the 2e7 run converged at 5)
PT2_FROM_ITER=1         # set to 4 to skip PT2 on the early, disk-hungry spaces

# ASCI
BATCH_SIZE=500
MERGE_MODE=include
EPS_THRESH=1e-10
ECONV=1e-4
ASCI_MAXDISK=3000       # GiB

# PT2
EPS2=1e-7
PBATCH=5000
COMPACT=50
PT2_MAXDISK=3300        # GiB, Julia-side abort
PT2_HARDKILL_GIB=3700   # GiB, script-side SIGKILL between compactions
# ==============================================================================

RATIO=$(awk -v c="$N_CDETS" -v t="$N_TDETS" 'BEGIN{printf "%.4f", c/t}')
echo "N_cdets / N_tdets = ${RATIO}   (series convention: 0.0500)"
awk -v r="$RATIO" 'BEGIN{ if (r < 0.045 || r > 0.055)
    print "*** WARNING: ratio is OFF-CONVENTION. This point will not be comparable. ***" }'

human() {
    local n=$1
    if   [ $((n % 1000000)) -eq 0 ]; then echo "$((n / 1000000))M"
    elif [ $((n % 1000))    -eq 0 ]; then echo "$((n / 1000))k"
    else echo "${n}"; fi
}
TAG="O3_asci_c$(human ${N_CDETS})_t$(human ${N_TDETS})"

RESULTS=${BASE_DIR}/chemistry_experiments/ozone_new_inputs/results
OUTFILE=${RESULTS}/asci/${TAG}.json
CKPT=${RESULTS}/checkpoints/asci__${TAG}.ckpt
ARCHIVE=${RESULTS}/checkpoints/iters_${TAG}
LOGDIR=${HOME}/o3_20M_logs_${TAG}
mkdir -p "$(dirname "$OUTFILE")" "${RESULTS}/checkpoints" "$ARCHIVE" "$LOGDIR"

PT2_HARDKILL_B=$(( PT2_HARDKILL_GIB * 1024 * 1024 * 1024 ))

echo "=============================================================="
echo " tag        : ${TAG}"
echo " checkpoint : ${CKPT}"
echo " archive    : ${ARCHIVE}"
echo " logs       : ${LOGDIR}"
echo " eps2       : ${EPS2}   PT2 from iteration ${PT2_FROM_ITER}"
echo "=============================================================="
df -h /scratch/nh2429 | tail -1

# ---- one PT2 on a given checkpoint -------------------------------------------
run_pt2 () {
    local ck="$1" k="$2"
    local psp=/scratch/nh2429/pt2_${TAG}_it${k}_${SLURM_JOB_ID}
    local pkf; pkf=$(mktemp)
    rm -rf "$psp"; mkdir -p "$psp"

    "$JB" -t "$JULIA_NUM_THREADS" "$SD/run_asci_pt2_ooc_eps2.jl" \
        $NORB_TOT $NELEC_TOT $NORB_FRZN "$EPSILON" \
        "$PT2_DIR/eris.txt" "$PT2_DIR/hcore.txt" "$PT2_DIR/symm.txt" \
        "$ck" 0 "$PBATCH" "$psp" "$COMPACT" "$PT2_MAXDISK" "$EPS2" \
        > "${LOGDIR}/pt2_iter${k}.log" 2>&1 &
    local jpid=$!

    # The Julia MAXDISK check only fires at compaction points, so guard here too.
    ( while kill -0 "$jpid" 2>/dev/null; do
          local b; b=$(du -sb "$psp" 2>/dev/null | cut -f1)
          if [[ -n "$b" ]]; then
              echo "$b" >> "$pkf"
              awk -v b="$b" -v k="$k" \
                  'BEGIN{printf "  [it %s] SPILL %.1f GiB\n", k, b/1073741824}'
              if [[ "$b" -gt "$PT2_HARDKILL_B" ]]; then
                  echo "  *** HARD KILL at iteration $k: exceeds ${PT2_HARDKILL_GIB} GiB ***"
                  kill -9 "$jpid" 2>/dev/null; break
              fi
          fi
          sleep 120
      done ) & local mon=$!

    wait "$jpid"; local rc=$?
    kill "$mon" 2>/dev/null; wait "$mon" 2>/dev/null
    local peak; peak=$(sort -n "$pkf" 2>/dev/null | tail -1); [[ -z "$peak" ]] && peak=0
    awk -v p="$peak" -v k="$k" -v rc="$rc" \
        'BEGIN{printf "  PT2 iter %s: exit=%d  MEASURED PEAK = %.1f GiB\n", k, rc, p/1073741824}'
    rm -rf "$psp" "$pkf"
    return $rc
}

# ---- main loop ---------------------------------------------------------------
for (( k=1; k<=MAX_ITER; k++ )); do
    ARC="${ARCHIVE}/asci__${TAG}_iter${k}.ckpt"
    echo ""
    echo "########## ITERATION ${k} ##########"

    if [[ -f "$ARC" ]]; then
        echo "  ASCI iteration ${k}: archive exists, skipping."
    else
        RESUME=true; [[ $k -eq 1 ]] && RESUME=false
        [[ -f "$CKPT" ]] || RESUME=false
        SPILL=/scratch/nh2429/spill/asci_${SLURM_JOB_ID}_it${k}
        rm -rf "$SPILL"; mkdir -p "$SPILL"

        "$JB" --threads "$JULIA_NUM_THREADS" "$SD/run_asci.jl" \
            $NORB_TOT $NELEC_TOT $NORB_FRZN "$EPSILON" \
            "$ERI" "$HCORE" "$SYMM" \
            $N_CDETS $N_TDETS $BATCH_SIZE $k \
            "$OUTFILE" "$SPILL" \
            $MERGE_MODE $EPS_THRESH $ECONV $RESUME $ASCI_MAXDISK \
            2>&1 | tee "${LOGDIR}/asci_iter${k}.log"
        rc=${PIPESTATUS[0]}
        rm -rf "$SPILL"
        echo "  ASCI iteration ${k}: exit=${rc}"
        [[ $rc -ne 0 ]] && { echo "  ASCI FAILED -- stopping."; break; }

        cp "$CKPT" "$ARC"
        grep -E "Energy estimate|Converged" "${LOGDIR}/asci_iter${k}.log" | tail -2
    fi

    # Stop cleanly once ASCI reports convergence.
    CONVERGED=0
    grep -q "Converged" "${LOGDIR}/asci_iter${k}.log" 2>/dev/null && CONVERGED=1

    if [[ $k -ge $PT2_FROM_ITER ]]; then
        if grep -q "E_PT2 =" "${LOGDIR}/pt2_iter${k}.log" 2>/dev/null; then
            echo "  PT2 iteration ${k}: already done, skipping."
        else
            echo "  PT2 iteration ${k} at eps2 = ${EPS2} ..."
            # A PT2 failure (almost always a disk abort on an early, dense
            # space) must NOT kill the run -- the variational chain continues.
            run_pt2 "$ARC" "$k" || echo "  PT2 iteration ${k} FAILED; continuing."
        fi
    else
        echo "  PT2 iteration ${k}: below PT2_FROM_ITER=${PT2_FROM_ITER}, skipped."
    fi

    [[ $CONVERGED -eq 1 ]] && { echo ""; echo "ASCI converged at iteration ${k}."; break; }
done

# ---- summary -----------------------------------------------------------------
echo ""
echo "=============================================================="
echo " iter   E_var                 E_PT2                 E_var + E_PT2"
for (( k=1; k<=MAX_ITER; k++ )); do
    ev=$(grep -h "Energy estimate:" "${LOGDIR}/asci_iter${k}.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    ep=$(grep -h "E_PT2  *=" "${LOGDIR}/pt2_iter${k}.log" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')
    [[ -z "$ev" ]] && continue
    if [[ -n "$ep" ]]; then
        awk -v k="$k" -v a="$ev" -v b="$ep" \
            'BEGIN{printf " %4s   %-20s  %-20s  %.10f\n", k, a, b, a+b}'
    else
        printf " %4s   %-20s  %-20s  %s\n" "$k" "$ev" "(none)" "-"
    fi
done
echo "=============================================================="
echo "Resubmit this same script to continue; completed iterations are skipped."
