#!/bin/bash
#SBATCH --job-name=O3_greedy_1M
#SBATCH --output=O3_greedy_1M_%j.out
#SBATCH --error=O3_greedy_1M_%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --account=torch_pr_554_general

# ------------------------------------------------------------------------------
# Ozone, Greedy algorithm with truncation parameter M, OOC sharded-EXACT drain.
# matvec done with no filtration
#
# To resume from a checkpoint (if it exists):
#   sbatch --export=ALL,RESUME=true greedy_ooc.sbatch
#
# To restart fresh (delete the checkpoint first):
#   rm /scratch/nh2429/greedysparse/chemistry_experiments/ozone_new_inputs/results/checkpoints/greedy__O3_greedy_m1M_ooc.ckpt
#   sbatch greedy_ooc.sbatch
#
# SPILL_DIR needs to have enough space (>1TB for M = 1M), cleaned up on exit.
# ------------------------------------------------------------------------------

module purge
ulimit -n 65536 || ulimit -n 32768 || true
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

BASE_DIR=/scratch/nh2429/greedysparse
SCRIPT=${BASE_DIR}/chemistry_experiments/scripts/run_greedy_ooc.jl

# Integral files are produced by chemistry_experiments/ozone_new_inputs/input_data/TODO
O3_DIR=${BASE_DIR}/chemistry_experiments/ozone_new_inputs/input_data/asci_gsno_om
ERI_FILE=${O3_DIR}/eris.txt
HCORE_FILE=${O3_DIR}/hcore.txt
SYMM_FILE=${O3_DIR}/symm.txt

for f in "${ERI_FILE}" "${HCORE_FILE}" "${SYMM_FILE}"; do
    [[ -s "$f" ]] || { echo "MISSING OR EMPTY INPUT: $f"; exit 1; }
done

RESULTS_ROOT=${BASE_DIR}/chemistry_experiments/ozone_new_inputs/results
OUTFILE=${RESULTS_ROOT}/greedy/O3_greedy_m1M_ooc.json
mkdir -p $(dirname ${OUTFILE})
mkdir -p ${RESULTS_ROOT}/checkpoints

SPILL_DIR=/scratch/nh2429/spill/greedy_${SLURM_JOB_ID}
mkdir -p ${SPILL_DIR}
echo "Spill dir: ${SPILL_DIR}"
df -h ${SPILL_DIR}

cleanup() {
    echo "Cleaning up spill dir: ${SPILL_DIR}"
    rm -rf ${SPILL_DIR}
}
trap cleanup EXIT

NORB_TOT=90
NELEC_TOT=24
NORB_FRZN=3
EPSILON=1

M=1000000
BATCH_SIZE=500
Q_MAX=20

# RESUME flag — false unless overridden via --export=ALL,RESUME=true
RESUME=${RESUME:-false}
echo "RESUME = ${RESUME}"

/home/nh2429/julia/julia --threads ${JULIA_NUM_THREADS} \
    ${SCRIPT} \
    ${NORB_TOT} ${NELEC_TOT} ${NORB_FRZN} ${EPSILON} \
    ${ERI_FILE} ${HCORE_FILE} ${SYMM_FILE} \
    ${M} ${BATCH_SIZE} ${Q_MAX} \
    ${OUTFILE} \
    ${SPILL_DIR} \
    ${RESUME}
