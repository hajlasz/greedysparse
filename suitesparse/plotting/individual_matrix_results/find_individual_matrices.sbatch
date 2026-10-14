#!/bin/bash
#SBATCH --job-name=find_mats
#SBATCH --output=find_mats.out
#SBATCH --error=find_mats.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --account torch_pr_554_general

cd /scratch/nh2429/greedysparse/suitesparse/plotting/individual_matrix_results/

module purge
module load matlab/2025b

matlab -nodisplay -nosplash -nodesktop -r 'find_individual_matrices("diag")'