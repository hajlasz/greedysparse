#!/bin/bash
#SBATCH --job-name=O3_asci_20M
#SBATCH --output=O3_asci_20M_%j.out
#SBATCH --error=O3_asci_20M_%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=600G
#SBATCH --account=torch_pr_554_general
# ==============================================================================
# VARIATIONAL ASCI at |S| = 2e7, N_cdets = 1e6  (ratio 0.05, series convention).
#
# WHY. The 1e7 point misses chemical accuracy by 0.05 mHa against the
# 5-point benchmark a = -293.1812248. The identity
#
#     |a - y|  =  b/(1+b) * (E_var - a),      b = 0.071127
#
# says passing needs E_var <= -293.1572262, i.e. 1.2 mHa more variational
# energy than 1e7 delivered. Fitting the measured deficit (a - E_var) over all
# six sizes gives (a - E_var) ~ |S|^-0.2722, so
#
#     |S|       E_var (pred)     deficit    |a-y| (pred)
#     1e7      -293.1551530      26.07      1.731  (measured: 1.645)
#     2e7      -293.1596355      21.59      1.434  PASS
#     3e7      -293.1618912      19.33      1.284
#
# Crossover is at |S| ~ 1.36e7, so 2e7 clears it with ~0.16 mHa of margin.
# That margin is thin -- see the WARNING at the bottom.
#
# ------------------------------------------------------------------------------
# SIZING (from job 14780600 at |S| = 1e7, 16 threads)
#   search+merge   0.0319 s/generator x 1e6 gen  = 31,900 s
#   build sparse A 12600 s x (2)^1.22            = 29,400 s
#   ARPACK         2600 s x 2                    =  5,200 s
#   x 4.5 iterations                             ~ 83 h at 16 threads
#   -> 48 threads requested; expect ~35-45 h (Amdahl, not 3x)
#
#   nnz      = 2.16e10  (1080/row, from nnz/row ~ |S|^-0.15)
#   CSC RAM  = 322 GB at the measured 14.9 B/nnz
#   + vectors/ARPACK/determinants ~ 5 GB  ->  ~330 GB, inside 600G
#
#   nnz exceeds the Int32 limit (2.147e9) by 10x. The 1e7 run already needed
#   Int64 indices; confirm the build path is Int64 throughout before trusting
#   any result here -- a silent overflow would corrupt the matrix, not crash.
#
#   search spill ~ 2256 GiB at N_cdets = 1e6.  MAX_DISK_GB = 3000 below.
#   RUN THIS ALONE. /scratch is at 98%.
# ==============================================================================

set -o pipefail
module purge
ulimit -n 65536 || ulimit -n 32768 || true
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}

BASE_DIR=/scratch/nh2429/greedysparse
SCRIPT=${BASE_DIR}/chemistry_experiments/scripts/run_asci.jl
O3_DIR=/scratch/nh2429/greedy_sparse_clean/chemistry_experiments/ozone_new_inputs/input_data/asci_gsno_om

ERI_FILE=${O3_DIR}/eris.txt
HCORE_FILE=${O3_DIR}/hcore.txt
SYMM_FILE=${O3_DIR}/symm.txt
for f in "${ERI_FILE}" "${HCORE_FILE}" "${SYMM_FILE}"; do
    [[ -s "$f" ]] || { echo "MISSING OR EMPTY INPUT: $f"; exit 1; }
done

NORB_TOT=90
NELEC_TOT=24
NORB_FRZN=3
EPSILON=1e-3

N_CDETS=${N_CDETS:-1000000}
N_TDETS=${N_TDETS:-20000000}
BATCH_SIZE=500
Q_MAX=20
MERGE_MODE=include
EPS_THRESH=1e-10
ECONV=1e-4
MAX_DISK_GB=3000

RATIO=$(awk -v c="${N_CDETS}" -v t="${N_TDETS}" 'BEGIN{printf "%.4f", c/t}')
echo "N_cdets / N_tdets = ${RATIO}   (series convention: 0.0500)"
awk -v r="${RATIO}" 'BEGIN{ if (r < 0.045 || r > 0.055)
    print "*** WARNING: ratio is OFF-CONVENTION. This point will not be comparable. ***" }'

human() {
    local n=$1
    if   [ $((n % 1000000)) -eq 0 ]; then echo "$((n / 1000000))M"
    elif [ $((n % 1000))    -eq 0 ]; then echo "$((n / 1000))k"
    else echo "${n}"; fi
}
TAG="O3_asci_c$(human ${N_CDETS})_t$(human ${N_TDETS})"

RESULTS_ROOT=${BASE_DIR}/chemistry_experiments/ozone_new_inputs/results
OUTFILE=${RESULTS_ROOT}/asci/${TAG}.json
CKPT=${RESULTS_ROOT}/checkpoints/asci__${TAG}.ckpt
mkdir -p "$(dirname ${OUTFILE})" "${RESULTS_ROOT}/checkpoints"

SPILL_DIR=/scratch/nh2429/spill/asci_${SLURM_JOB_ID}
mkdir -p ${SPILL_DIR}

# Report BOTH disk and RSS. At this size either can be the binding constraint.
( while true; do
    R=$(ps -o rss= -C julia 2>/dev/null | sort -n | tail -1 | tr -d ' ')
    echo "[mon $(date +%H:%M)] spill $(du -sb ${SPILL_DIR} 2>/dev/null | cut -f1) B   RSS ${R:-0} kB"
    sleep 600
  done ) & MON_PID=$!
cleanup() { kill ${MON_PID} 2>/dev/null; echo "Cleaning spill: ${SPILL_DIR}"; rm -rf ${SPILL_DIR}; }
trap cleanup EXIT

RESUME=${RESUME:-false}

echo "=============================================================="
echo " run tag     : ${TAG}"
echo " checkpoint  : ${CKPT}"
echo " config      : N_cdets=${N_CDETS} N_tdets=${N_TDETS} threads=${JULIA_NUM_THREADS}"
echo " predicted   : E_var ~ -293.1596  (pass needs <= -293.1572262)"
echo "=============================================================="
df -h ${SPILL_DIR}

/home/nh2429/julia/julia --threads ${JULIA_NUM_THREADS} \
    ${SCRIPT} \
    ${NORB_TOT} ${NELEC_TOT} ${NORB_FRZN} ${EPSILON} \
    ${ERI_FILE} ${HCORE_FILE} ${SYMM_FILE} \
    ${N_CDETS} ${N_TDETS} ${BATCH_SIZE} ${Q_MAX} \
    "${OUTFILE}" "${SPILL_DIR}" \
    ${MERGE_MODE} ${EPS_THRESH} ${ECONV} ${RESUME} ${MAX_DISK_GB}
rc=$?
echo "ASCI exit: ${rc}"
echo "If this timed out, resubmit with --export=ALL,RESUME=true (checkpoint is written per iteration)."
exit ${rc}
