#!/bin/bash

#SBATCH --partition=bigmem
#SBATCH --job-name=MRMS_SOM_regions_py_SE_24hrQPE_3x2_thresh_ari1yr24hr_newdomains_sklearn_24hr_test
#SBATCH --account=wrfruc
#SBATCH --qos=batch
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --output=MRMS_SOM_regions_py_SE_24hrQPE_3x2_thresh_ari1yr24hr_newdomains_sklearn_24hr_test.out
#SBATCH --error=MRMS_SOM_regions_py_SE_24hrQPE_3x2_thresh_ari1yr24hr_newdomains_sklearn_24hr_test.err


###module load miniconda

source /scratch1/BMC/wrfruc/naegele/miniconda3/etc/profile.d/conda.sh
conda activate /scratch1/BMC/wrfruc/naegele/mySOM2/

###myregions=("NW" "NC" "NE" "SW" "SC" "SE")
precip_duration=("24hr") #1hr #24hr
myregions=("SE")
mySOMsize_dim1=3 #5 #3 #4 #5
mySOMsize_dim2=2 #5 #2 #3 #5
for myregion in ${myregions[@]};
do
    srun python -u SOM_test2_thresh_newdomains_sklearn.py $precip_duration $myregion $mySOMsize_dim1 $mySOMsize_dim2
###    srun python SOM_test2_multivar_thresh_part1.py 
done
###srun bash SOM_test2.sh
