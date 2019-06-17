#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N spkr_lit_badlex           
#$ -cwd                  
#$ -l h_rt=10:00:00 
#$ -l h_vmem=1G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda/5.0.1

# Run the program
source activate sim-prag
python level_prag_learn.py "/exports/eddie/scratch/s1682785/posts" 1 --spkr 171 --nruns 30
source deactivate