#!/bin/bash
#SBATCH --job-name=Ne_greedy
#SBATCH --output=logs/greedy/Ne_greedy_%A_%a.out
#SBATCH --error=logs/greedy/Ne_greedy_%A_%a.err
#SBATCH --array=0-19
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --account torch_pr_554_general

# -------------------------------
# Environment
# -------------------------------
module purge
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# -------------------------------
# Problem setup (Neon example)
# -------------------------------
NORB_TOT=23
NELEC_TOT=10
NORB_FRZN=1
EPSILON=1e-3

ERI_FILE=/scratch/nh2429/greedysparse/chemistry_experiments/Neon/input_data/eris.txt
HCORE_FILE=/scratch/nh2429/greedysparse/chemistry_experiments/Neon/input_data/hcore.txt
SYMM_FILE=/scratch/nh2429/greedysparse/chemistry_experiments/Neon/input_data/symm.txt

MAX_ITERS=500
REL_TOL=1e-11
EVAL_TRUE=128.709475548738

# -------------------------------
# Bag sizes
# -------------------------------
BAGSIZES=(1000 2000 4000 8000 16000 35000 70000 87500 105000 122500 140000 157500 175000 192500 210000 227500 245000 262500 280000)

K_KEEP=${BAGSIZES[$SLURM_ARRAY_TASK_ID]}

OUTDIR=/scratch/nh2429/greedysparse/chemistry_experiments/Neon/results/greedy_sparse
mkdir -p $OUTDIR logs

OUTFILE=$OUTDIR/greedy_k${K_KEEP}.json

echo "-------------------------------------"
echo "SLURM JOB ID      = $SLURM_JOB_ID"
echo "ARRAY TASK ID    = $SLURM_ARRAY_TASK_ID"
echo "K_KEEP           = $K_KEEP"
echo "THREADS          = $JULIA_NUM_THREADS"
echo "-------------------------------------"

# -------------------------------
# Run greedy
# -------------------------------
/home/nh2429/julia/julia --threads ${JULIA_NUM_THREADS} /scratch/nh2429/greedysparse/chemistry_experiments/scripts/run_greedy.jl \
    $NORB_TOT \
    $NELEC_TOT \
    $NORB_FRZN \
    $EPSILON \
    $ERI_FILE \
    $HCORE_FILE \
    $SYMM_FILE \
    $K_KEEP \
    $OUTFILE \
    $MAX_ITERS \
    $REL_TOL \
    $EVAL_TRUE
