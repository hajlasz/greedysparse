#!/bin/bash
#SBATCH --job-name=asci_5M
#SBATCH --output=asci_c250k_t5M_%j.out
#SBATCH --error=asci_c250k_t5M_%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --account=torch_pr_554_general

# ==============================================================================
# Ozone ASCI: out-of-core search, explicit sparse-CSC eigensolve.
# ------------------------------------------------------------------------------
# RESUME
# ------------------------------------------------------------------------------
# The checkpoint path is derived from OUTFILE by checkpoint_path_for:
#
#     .../ozone/results/asci/<TAG>.json
#         ->  .../ozone/results/checkpoints/asci__<TAG>.ckpt
#
# To continue an interrupted run:
#     sbatch --export=ALL,RESUME=true asci_ozone.sbatch
# To start fresh, delete that checkpoint first. The path is echoed below, so
# copy it from the log rather than retyping it.
#
# Checkpoints are interchangeable with the greedy and TPM drivers (shared
# format), and the PT2 scripts read exactly this file.
# ==============================================================================

set -o pipefail
module purge
ulimit -n 65536 || ulimit -n 32768 || true
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

BASE_DIR=/scratch/nh2429/greedysparse
SCRIPT=${BASE_DIR}/chemistry_experiments/scripts/run_asci.jl


O3_DIR=/scratch/nh2429/greedy_sparse_clean/chemistry_experiments/ozone_new_inputs/input_data/asci_gsno_om

ERI_FILE=${O3_DIR}/eris.txt
HCORE_FILE=${O3_DIR}/hcore.txt
SYMM_FILE=${O3_DIR}/symm.txt

# Fail in seconds rather than after a 48-hour queue wait plus a load attempt.
for f in "${ERI_FILE}" "${HCORE_FILE}" "${SYMM_FILE}"; do
    [[ -s "$f" ]] || { echo "MISSING OR EMPTY INPUT: $f"; exit 1; }
done

# ==============================================================================
# CONFIG  -- EDIT THIS BLOCK
# ==============================================================================
NORB_TOT=90        # spatial orbitals, including frozen
NELEC_TOT=24       # electrons, including frozen
NORB_FRZN=3        # frozen orbitals  (so the active space is CAS(87,18))
EPSILON=1e-3       # shift parameter in A = I - eps*H

N_CDETS=250000      # core size: how many determinants may generate candidates
N_TDETS=5000000     # target size: how many determinants to keep in the search
BATCH_SIZE=500
Q_MAX=20

MERGE_MODE=include # include = MACIS rule (generators never evicted); or rank
EPS_THRESH=1e-10   # search screen on the full partial score; 0 keeps everything
ECONV=1e-4

MAX_DISK_GB=4193  # GiB. Abort cleanly rather than fill shared /scratch.
# ==============================================================================

# Derive the run tag from the parameters so the two can never drift apart.
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

# The method subdirectory matters: checkpoint_path_for takes the method name
# from the PARENT DIRECTORY of the JSON file. A results file written straight
# into results/ would yield checkpoints/results__<TAG>.ckpt, which the PT2
# scripts would not find.
mkdir -p "$(dirname ${OUTFILE})"
mkdir -p "${RESULTS_ROOT}/checkpoints"

SPILL_DIR=/scratch/nh2429/spill/asci_${SLURM_JOB_ID}
mkdir -p ${SPILL_DIR}

( while true; do echo "[disk $(date +%F_%H:%M)] $(df -h ${SPILL_DIR} | tail -1)"; sleep 600; done ) &
MON_PID=$!

cleanup() {
    kill ${MON_PID} 2>/dev/null
    echo "Cleaning up spill dir: ${SPILL_DIR}"
    rm -rf ${SPILL_DIR}
}
trap cleanup EXIT

RESUME=${RESUME:-false}

echo "=============================================================="
echo " run tag     : ${TAG}"
echo " integrals   : ${O3_DIR}   (GSNO / CASSCF natural orbitals)"
echo " outfile     : ${OUTFILE}"
echo " checkpoint  : ${CKPT}"
echo " spill dir   : ${SPILL_DIR}"
echo " resume      : ${RESUME}"
echo " config      : N_cdets=${N_CDETS} N_tdets=${N_TDETS} batch=${BATCH_SIZE} q_max=${Q_MAX}"
echo "               merge=${MERGE_MODE} eps_thresh=${EPS_THRESH} econv=${ECONV} max_disk_gb=${MAX_DISK_GB}"
echo "=============================================================="
df -h ${SPILL_DIR}

if [[ "${RESUME}" == "true" && ! -f "${CKPT}" ]]; then
    echo "RESUME=true but no checkpoint at ${CKPT} -- starting fresh instead."
fi

/home/nh2429/julia/julia --threads ${JULIA_NUM_THREADS} \
    ${SCRIPT} \
    ${NORB_TOT} ${NELEC_TOT} ${NORB_FRZN} ${EPSILON} \
    ${ERI_FILE} ${HCORE_FILE} ${SYMM_FILE} \
    ${N_CDETS} ${N_TDETS} ${BATCH_SIZE} ${Q_MAX} \
    "${OUTFILE}" \
    "${SPILL_DIR}" \
    ${MERGE_MODE} ${EPS_THRESH} ${ECONV} ${RESUME} ${MAX_DISK_GB}
rc=$?
echo "ASCI exit: ${rc}"
exit ${rc}
