# Import necessary modules
import glob
import sys
import numpy as np
#from mysom2 import mysom2  # my own function for training the SOM
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import xarray as xr
import pickle
from datetime import datetime
import seaborn as sns
from minisom import MiniSom

#myregion_plot = str(sys.argv[1])
#print('My region =', myregion_plot)
myregion_plot = 'SC'
SOMsize_str = '5x5'
ari = '1yr'
precip_duration1 = '1hr'
precip_duration2 = '24hr'
varname_pickle = 'radaronly_QPE_01H_24H'

# Pickle files as of 03 May 2024
#-rw-r--r-- 1 Steven.Naegele wrfruc 6.2M Apr 16 19:12 SOM_5x5_multivar_SC_precip_thresh_count_ari10yr1hr.pkl
#-rw-r--r-- 1 Steven.Naegele wrfruc 6.2M Apr 18 20:40 SOM_5x5_multivar_SC_precip_thresh_count_ari1yr.pkl
#-rw-r--r-- 1 Steven.Naegele wrfruc 635M May  1 17:43 SOM_5x5_thresh_ari1yr1hr_ari1yr24hr_radaronly_QPE_01H_24H_SC_vars_batch.pkl
#with open('SOM_3x2_MultiSensor_QPE_01H_Pass2_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
##with open('SOM_4x3_MultiSensor_QPE_01H_Pass2_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
##    varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles, \
#    varSOM, dataArray, varSOM_nonzero, dataArray_nonzero, map_lats, map_lons, months_doubles, years_doubles, \
#    year_month_strs, year_month_counts, mapedges, myfiles, varname, SOMsize, mapSize, myregion = pickle.load(f)
#with open('SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
#    varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars, year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapedges, myregion = pickle.load(f)
with open('SOM_' + SOMsize_str + '_multivar_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
    varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars, year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapedges, myregion = pickle.load(f)

varSOM_dummy = varSOM
#mypath1 = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_01H_Pass2/20210101_20231231/'
mypath1 = '/scratch1/BMC/wrfruc/ejames/climo/mrms/01h/**/'
#mypath2 = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_24H_Pass2/20210101_20231231/'
#mypaths = [mypath1, mypath2]
#myfilenames1 = 'MultiSensor_QPE_01H_Pass2_00.00_202*0000.nc'
myfilenames1 = 'mrms_radaronly_20*.nc'
#myfilenames2 = 'MultiSensor_QPE_24H_Pass2_00.00_2021060*0000.nc'
#myfilenames = [myfilenames1, myfilenames2]
#myfiles = []
mydatetimes = []
#nMaps = []
#filename_firsts = []
#varnames = []
#varname_titles = []
#date_firsts = []
#filename_lasts = []
#date_lasts = []
# for myfile in sorted(glob.glob(mypath + 'MultiSensor_QPE_01H_Pass2_00.00_2021022*.grib2')):
#for myfile in sorted(glob.glob(mypath + 'MultiSensor_QPE_01H_Pass2_00.00_20*0000.nc')):
#for myvar in range(0, len(mypaths)):
#myfiles.append(sorted(glob.glob(mypath1 + myfilenames1)))
myfiles = sorted(glob.glob(mypath1 + myfilenames1))
#print(myfiles)
#myfiles2 = []
#for myfile2 in sorted(glob.glob(mypath2 + 'MultiSensor_QPE_24H_Pass2_00.00_20*0000.nc')):
#    myfiles2.append(myfile2)

#    nMaps.append(len(myfiles[myvar]))  # 26240, 26235
#print(nMaps1)  # 26240
#print(nMaps2)  # 26235
#    example_full_filename = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_01H_Pass2/20210101_20231231/MultiSensor_QPE_01H_Pass2_00.00_20210101-000000.nc'

#    sample_filename = 'MultiSensor_QPE_01H_Pass2_00.00_20210101-000000.nc'
for myfile in myfiles:
#    filename_first = myfiles[myvar][0]
#    filename_firsts.append(filename_first)
#    varname = filename_first.split('/')[-1].split('.')[0][0:-3]
#    varnames.append(varname)
#    varname_title = varname.replace('_',' ')
#    varname_titles.append(varname_title)
    mydatetime = myfile.split('/')[-1].split('.')[1][3:13+1]
#    mydatetime = datetime.datetime.strptime(mydatetime, '%Y%m%d-%H')
    mydatetime = mydatetime[0:3+1] + '-' + mydatetime[4:5+1] + '-' + mydatetime[6:7+1] + ' ' + mydatetime[9:10+1] + ':00:00'
    mydatetimes.append(mydatetime)
#    date_firsts.append(date_first)
#    filename_last = myfiles[myvar][-1]
#    filename_lasts.append(filename_last)
#    date_last = filename_last.split('/')[-1].split('.')[1][3:10+1]
#    date_last = datetime.datetime.strptime(date_last, '%Y%m%d')
#    date_lasts.append(date_last)
#print(mydatetimes)
#print(type(dataArray))

winning_nodes = varSOM.win_map(dataArray_train, return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
                                                                # note that the SOM node indices are [col, row], and I will read through cols first

for iSOM in range(0, SOMsize[1]):  # row
    for jSOM in range(0, SOMsize[0]):  # col
        winning_node_indices = winning_nodes[jSOM, iSOM]
        print('node: ' + str(jSOM) + ', ' + str(iSOM))
        print('len(winning_node_indices) = ' + str(len(winning_node_indices)))
        print(winning_node_indices)
        print(np.array(mydatetimes)[winning_node_indices])


