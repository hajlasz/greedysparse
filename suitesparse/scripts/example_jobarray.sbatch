#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=GREEDY_small
#SBATCH --output=GREEDY_small%a.out
#SBATCH --array=0-11
#SBATCH --account=torch_pr_554_general

cd /scratch/nh2429/greedysparse/suitesparse/scripts/

module purge
module load matlab/2025b

ind_start=("11"   "632" "717" "797" "840" "882" "906" "931" "934" "952" "976" "998")
ind_end=(  "631" "716" "796" "839" "881" "905" "930" "933" "951" "975" "997" "1106")

matlab -nodisplay -nosplash -nodesktop -r "test_greedy(${ind_start[$SLURM_ARRAY_TASK_ID]}, ${ind_end[$SLURM_ARRAY_TASK_ID]})"