#!/bin/bash
#SBATCH --job-name=O3_pt2_20M
#SBATCH --output=O3_pt2_20M_%j.out
#SBATCH --error=O3_pt2_20M_%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48
#SBATCH --mem=600G --time=48:00:00
#SBATCH --account=torch_pr_554_general
# ==============================================================================
# SCREENED PT2 at |S| = 2e7.  Two-value ladder: eps2 = 2e-7 then 1.5e-7.
#
# WHY TWO VALUES, AND WHY NOT 1e-7.
# Disk scales as eps2^-1.387 (measured at 1e7: 1030.2 GiB at 2e-7 -> 2694.6 GiB
# at 1e-7) and as |S|^0.5 at fixed eps2 (externals/source ~ |S|^-0.5). At 2e7:
#
#     eps2      projected peak      vs 3800 GiB
#     2.0e-7        1457               0.38x
#     1.5e-7        2154               0.57x
#     1.2e-7        2943               0.77x
#     1.0e-7        3811               1.00x   <-- DOES NOT FIT
#
# So 1e-7 is out at this size. 2e-7 matches the five benchmark points exactly
# (uniform protocol); 1.5e-7 gives the convergence gap.
#
# BIAS ESTIMATE. The 8-point ladder at 1e7 finally stabilised beta at 2.708,
# so the successive-gap estimator is valid:
#
#     residual bias  ~=  |gap| / (r^beta - 1),   r = 2/1.5 = 1.333  ->  r^beta = 2.20
#
# i.e. bias at 1.5e-7 ~= gap/1.20. For reference, at 1e7 the measured total
# correction from 2e-7 to eps2->0 was only 0.094 mHa, and the earlier 0.18 mHa
# bound was conservative by more than 2x.
#
# REPORT THE VERDICT TWICE: once at eps2 = 2e-7 (uniform with the benchmark
# points) and once eps2-extrapolated. Both should pass if the prediction holds;
# the eps2 correction moves |E_PT2| UP, i.e. toward passing.
#
# WALL: ~4.2 h at 2e-7, ~6 h at 1.5e-7, plus reduce. 48 h is ample.
# RUN ALONE. /scratch is at 98%.
# ==============================================================================
set -o pipefail
ulimit -n 65536 || ulimit -n 32768 || true
module purge
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
JB=/home/nh2429/julia/julia

BASE_DIR=/scratch/nh2429/greedysparse
SD=${BASE_DIR}/chemistry_experiments/scripts
BASE=${BASE_DIR}/chemistry_experiments/ozone_new_inputs/input_data/asci_gsno_om

N_CDETS=${N_CDETS:-1000000}
N_TDETS=${N_TDETS:-20000000}

POUT=0
PBATCH=5000
COMPACT=50
MAXDISK=3400
HARDKILL_GIB=3600
EPS=1e-3
EPS2_LIST=${EPS2_LIST:-"2e-7 1.5e-7"}

human() {
    local n=$1
    if   [ $((n % 1000000)) -eq 0 ]; then echo "$((n / 1000000))M"
    elif [ $((n % 1000))    -eq 0 ]; then echo "$((n / 1000))k"
    else echo "${n}"; fi
}
TAG="O3_asci_c$(human ${N_CDETS})_t$(human ${N_TDETS})"
CK=${BASE_DIR}/chemistry_experiments/ozone_new_inputs/results/checkpoints/asci__${TAG}.ckpt

HARDKILL_B=$(( HARDKILL_GIB * 1024 * 1024 * 1024 ))
[[ -f "$CK" ]] || { echo "no checkpoint: $CK"; echo "run asci_20M.sbatch first"; exit 1; }
echo "checkpoint : $CK"
echo "eps2 ladder: $EPS2_LIST"
echo "expect E_var ~ -293.1596 (pass needs <= -293.1572262)"

for E2 in $EPS2_LIST; do
    PSP=/scratch/nh2429/pt2_20M_${SLURM_JOB_ID}_${E2}
    PKF=$(mktemp)
    rm -rf "$PSP"; mkdir -p "$PSP"
    echo ""
    echo "=============================================================="
    echo "  eps2 = $E2    spill = $PSP"
    echo "=============================================================="
    df -h "$PSP"

    t0=$(date +%s)
    "$JB" -t "$JULIA_NUM_THREADS" "$SD/run_asci_pt2_ooc_eps2.jl" \
      90 24 3 "$EPS" "$BASE/eris.txt" "$BASE/hcore.txt" "$BASE/symm.txt" \
      "$CK" "$POUT" "$PBATCH" "$PSP" "$COMPACT" "$MAXDISK" "$E2" \
      2> "$HOME/pt2_20M_${SLURM_JOB_ID}_${E2}.err" &
    JPID=$!

    ( while kill -0 "$JPID" 2>/dev/null; do
        B=$(du -sb "$PSP" 2>/dev/null | cut -f1)
        R=$(ps -o rss= -p "$JPID" 2>/dev/null | tr -d ' ')
        if [[ -n "$B" ]]; then
            echo "$B" >> "$PKF"
            awk -v b="$B" -v r="${R:-0}" -v e="$E2" \
                'BEGIN{printf "SPILL[%s] %.2f GiB   RSS %.1f GB\n", e, b/1073741824, r/1048576}'
            if [[ "$B" -gt "$HARDKILL_B" ]]; then
                echo "*** HARD KILL: exceeds ${HARDKILL_GIB} GiB ***"
                kill -9 "$JPID" 2>/dev/null; break
            fi
        fi
        sleep 60
      done ) & MON=$!

    wait "$JPID"; rc=$?
    kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null
    t1=$(date +%s)
    PEAK_B=$(sort -n "$PKF" 2>/dev/null | tail -1); [[ -z "$PEAK_B" ]] && PEAK_B=0
    awk -v p="$PEAK_B" -v w="$((t1-t0))" -v rc="$rc" -v e="$E2" \
        'BEGIN{printf "RESULT eps2=%s  exit=%d  wall=%ds (%.2f h)  MEASURED PEAK = %.1f GiB\n",
               e, rc, w, w/3600, p/1073741824}'
    rm -rf "$PSP" "$PKF"
    [[ $rc -ne 0 ]] && { echo "ABORTING LADDER at eps2=$E2"; exit $rc; }
done

echo ""
echo "LADDER COMPLETE."
echo "  bias at 1.5e-7 ~= |E(2e-7) - E(1.5e-7)| / 1.20   (beta = 2.708 from the 1e7 ladder)"
echo "  y_20M = E_var + E_PT2 ; compare |a - y| to 1.594 mHa, a = -293.1812248"
