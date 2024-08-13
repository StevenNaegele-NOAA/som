#!/bin/bash

#SBATCH --partition=bigmem
#SBATCH --job-name=MRMS_SOM_regions_nc_py_SE_multivar_3x2_thresh_ari1yr1hr_ari1yr24hr_newdomains_sklearn_noARI_2021_lr975_6hr
#SBATCH --account=wrfruc
#SBATCH --qos=batch
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --output=MRMS_SOM_regions_nc_py_SE_multivar_3x2_thresh_ari1yr1hr_ari1yr24hr_newdomains_sklearn_noARI_2021_lr975_6hr.out
#SBATCH --error=MRMS_SOM_regions_nc_py_SE_multivar_3x2_thresh_ari1yr1hr_ari1yr24hr_newdomains_sklearn_noARI_2021_lr975_6hr.err


###module load miniconda

source /scratch1/BMC/wrfruc/naegele/miniconda3/etc/profile.d/conda.sh
conda activate /scratch1/BMC/wrfruc/naegele/mySOM2/

###myregions=("NW" "NC" "NE" "SW" "SC" "SE")
myregions=("SE")
mySOMsize_dim1=3 #5 #3 #4 #5
mySOMsize_dim2=2 #5 #2 #3 #5
for myregion in ${myregions[@]};
do
    srun python SOM_test2_multivar_thresh_newdomains_sklearn_noARI.py $myregion $mySOMsize_dim1 $mySOMsize_dim2
###    srun python SOM_test2_multivar_thresh_part1.py 
done
###srun bash SOM_test2.sh
