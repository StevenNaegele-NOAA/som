#!/bin/bash

#SBATCH --partition=bigmem
#SBATCH --job-name=precip_thresh_count_SE_24hrQPE_3x2_thresh_ari1yr24hr_24hr
#SBATCH --account=wrfruc
#SBATCH --qos=batch
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --output=precip_thresh_count_SE_24hrQPE_3x2_thresh_ari1yr24hr_24hr.out
#SBATCH --error=precip_thresh_count_SE_24hrQPE_3x2_thresh_ari1yr24hr_24hr.err


###module load miniconda

source /scratch1/BMC/wrfruc/naegele/miniconda3/etc/profile.d/conda.sh
conda activate /scratch1/BMC/wrfruc/naegele/mySOM2/

###myregions=("NW" "NC" "NE" "SW" "SC" "SE")
###myregions=("SC")
###mySOMsize_dim1=5
###mySOMsize_dim2=5
###for myregion in ${myregions[@]};
###do
###    srun python SOM_test2_multivar_thresh.py $myregion $mySOMsize_dim1 $mySOMsize_dim2
###done
srun python precip_thresh_count_alt_6hr_onevar.py 
###srun bash SOM_test2.sh
