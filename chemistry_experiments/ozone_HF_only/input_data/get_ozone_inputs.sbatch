#!/bin/bash

#SBATCH --job-name=OZONE
#SBATCH --output=OZONE_get_inputs.out
#SBATCH --mem=50G
#SBATCH --time=06:00:00

module purge

python get_ozone_inputs.py