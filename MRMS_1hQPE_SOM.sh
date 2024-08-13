#!/bin/bash -l
#
# -- Request that this job run on hera
#SBATCH --partition=hera
#
# -- Request 16 cores
#SBATCH --ntasks=16
#
# -- Specify a maximum wallclock of 4 hours
#SBATCH --time=4:00:00
#
# -- Specify under which account a job should run
#SBATCH --account=wrfruc
#
# -- Set the name of the job, or Slurm will default to the name of the script
#SBATCH --job-name=MRMS_1hQPE_SOM_test
#
# -- Tell the batch system to set the working directory to the current working directory
#SBATCH --chdir=.

nt=$SLURM_NTASKS

conda activate SOM
srun -n $nt ./MRMS_1hQPE_SOM.py
