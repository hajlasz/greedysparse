#!/bin/bash
#SBATCH --job-name=pt2_200k
#SBATCH --output=pt2_200k_%j.out
#SBATCH --error=pt2_200k_%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=150G --time=12:00:00
#SBATCH --account torch_pr_554_general
# ------------------------------------------------------------------------------
# Ozone EN-PT2, EXACT (no eps_2 screening), on the |S| = 2e5 checkpoint.
# Uses the compacting OOC build (asci_pt2_ooc_compact), so peak DISK tracks the
# number of UNIQUE externals rather than the number of contributions.
# ------------------------------------------------------------------------------
set -o pipefail
ulimit -n 65536 || ulimit -n 32768 || true      # OOC merge opens ~1 fd per run per shard
module purge
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
JB=/home/nh2429/julia/julia

# Code and integrals live in DIFFERENT trees. Both paths below are copied from
# ozone_new_inputs/sbatch/asci_sbatch/asci.sbatch, which is what produced the
# checkpoint. Do not "tidy" them into one root.
SD=/scratch/nh2429/greedysparse/chemistry_experiments/scripts
BASE=/scratch/nh2429/greedysparse/chemistry_experiments/ozone_new_inputs/input_data/asci_gsno_om

# ===== EDIT =====
TAG=1M
POUT=0            # 0 -> internal default max(64, 2*threads); keep MODEST (fd budget)
PBATCH=500
COMPACT=200       # compact each shard every COMPACT batches (bounds disk)
MAXDISK=2000      # GiB. Abort cleanly rather than fill shared /scratch.
EPS=1e-3          # cancels out of PT2; any nonzero value is fine

# COPY THIS FROM THE ASCI JOB'S BANNER ("checkpoint  : ..."). The value below is
# the correct path to the checkpoint file.
CK=/scratch/nh2429/greedysparse/chemistry_experiments/ozone_new_inputs/results/checkpoints/asci__O3_asci_c500k_t1M.ckpt
# ================

PSP=/scratch/nh2429/pt2_ooc_spill_${TAG}_${SLURM_JOB_ID}
[[ -f "$CK" ]] || { echo "no checkpoint: $CK"; exit 1; }

# Clean the spill dir on ANY exit (success, failure, timeout). /scratch is
# shared and stale TB-scale files would eat the quota.
cleanup() { echo "cleanup: rm -rf $PSP"; rm -rf "$PSP"; }
trap cleanup EXIT
rm -rf "$PSP"; mkdir -p "$PSP"
echo "checkpoint: $CK"
echo "spill dir : $PSP"; df -h "$PSP"

# Background disk monitor so you can watch it plateau.
( while true; do echo "SPILL $(du -sb "$PSP" 2>/dev/null | cut -f1)"; sleep 120; done ) & MON=$!

# Route Julia's stderr to $HOME (NOT /scratch): if the disk ever fills, the
# traceback still lands somewhere readable.
"$JB" -t "$JULIA_NUM_THREADS" "$SD/run_asci_pt2_ooc.jl" \
  90 24 3 "$EPS" "$BASE/eris.txt" "$BASE/hcore.txt" "$BASE/symm.txt" \
  "$CK" "$POUT" "$PBATCH" "$PSP" "$COMPACT" "$MAXDISK" \
  2> "$HOME/pt2_ooc_${TAG}_${SLURM_JOB_ID}.err"
rc=$?
kill $MON 2>/dev/null
echo "PT2 OOC exit: $rc"
exit $rc
