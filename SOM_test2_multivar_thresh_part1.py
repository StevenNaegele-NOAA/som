# This script will create a SOM trained on MRMS MultiSensor 01H QPE

# Import necessary modules
import glob
import sys
import numpy as np
from mysom2_multivar_thresh_part1 import mysom2_multivar_thresh_part1  # my own function for training the SOM on multiple variables, split in two to avoid time limit exceedence
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import xarray as xr
import datetime

ari = '1yr'
precip_duration1 = '1hr'
precip_duration2 = '24hr'

print('Reading in MRMS data ...')
#mypath = '/Users/steven.naegele/Data/MRMS/20210101_20231231/'
#mypath = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_01H_Pass2/20210101_20231231/'
#mypath1 = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_01H_Pass2/20210101_20231231/'
#mypath2 = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_24H_Pass2/20210101_20231231/'
mypath1 = '/scratch1/BMC/wrfruc/ejames/climo/mrms/01h/2021*/'
mypath2 = '/scratch1/BMC/wrfruc/ejames/climo/mrms/24h/2021*/'
mypaths = [mypath1, mypath2]
#myfilenames1 = 'MultiSensor_QPE_01H_Pass2_00.00_202*0000.nc'
#myfilenames2 = 'MultiSensor_QPE_24H_Pass2_00.00_202*0000.nc'
#myfilenames = [myfilenames1, myfilenames2]
myfilenames = 'mrms_radaronly_20*.nc'
myfiles = []
nMaps = []
filename_firsts = []
varnames = []
varname_titles = []
date_firsts = []
date_firsts_strs = []
filename_lasts = []
date_lasts = []
date_lasts_strs = []
# for myfile in sorted(glob.glob(mypath + 'MultiSensor_QPE_01H_Pass2_00.00_2021022*.grib2')):
#for myfile in sorted(glob.glob(mypath + 'MultiSensor_QPE_01H_Pass2_00.00_20*0000.nc')):
for myvar in range(0, len(mypaths)):
#    myfiles.append(sorted(glob.glob(mypaths[myvar] + myfilenames[myvar])))
    myfiles.append(sorted(glob.glob(mypaths[myvar] + myfilenames)))
#myfiles2 = []
#for myfile2 in sorted(glob.glob(mypath2 + 'MultiSensor_QPE_24H_Pass2_00.00_20*0000.nc')):
#    myfiles2.append(myfile2)

    nMaps.append(len(myfiles[myvar]))  # 26240, 26235
    print('len(myfiles[myvar]) = ' + str(len(myfiles[myvar])))  # 
#print(nMaps1)  # 26240
#print(nMaps2)  # 26235
#    example_full_filename = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_01H_Pass2/20210101_20231231/MultiSensor_QPE_01H_Pass2_00.00_20210101-000000.nc'
    example_full_filename = '/scratch1/BMC/wrfruc/ejames/climo/mrms/01h/202212/mrms_radaronly_2022123123.nc'

#    sample_filename = 'MultiSensor_QPE_01H_Pass2_00.00_20210101-000000.nc'
    filename_first = myfiles[myvar][0]
    filename_firsts.append(filename_first)
#    varname = filename_first.split('/')[-1].split('.')[0][0:-3]
    varname = "_".join(filename_first.split("_", 2)[:2])  # split the filename up to the second occurrence of "_" and combine the first two elements
    varnames.append(varname)
    varname_title = varname.replace('_',' ')
    varname_titles.append(varname_title)
#    date_first = filename_first.split('/')[-1].split('.')[1][3:10+1]
    date_first = filename_first.split('_')[-1][0:7+1]
    date_firsts_strs.append(date_first)
    date_first = datetime.datetime.strptime(date_first, '%Y%m%d')
    date_firsts.append(date_first)
    filename_last = myfiles[myvar][-1]
    filename_lasts.append(filename_last)
#    date_last = filename_last.split('/')[-1].split('.')[1][3:10+1]
    date_last = filename_last.split('_')[-1][0:7+1]
    date_lasts_strs.append(date_last)
    date_last = datetime.datetime.strptime(date_last, '%Y%m%d')
    date_lasts.append(date_last)
#filename_first2 = myfiles2[0]
#varname2 = filename_first2.split('/')[-1].split('.')[0][0:-3]
#varname_title2 = varname2.replace('_',' ')
#date_first2 = filename_first2.split('/')[-1].split('.')[1][3:10+1]
#date_first2 = datetime.datetime.strptime(date_first2, '%Y%m%d')
#filename_last2 = myfiles2[-1]
#date_last2 = filename_last2.split('/')[-1].split('.')[1][3:10+1]
#date_last2 = datetime.datetime.strptime(date_last2, '%Y%m%d')
if date_firsts[0] < date_firsts[1]:
    date_first = date_firsts[0]
    date_first_str = date_firsts_strs[0]
elif date_firsts[0] >= date_firsts[1]:
    date_first = date_firsts[1]
    date_first_str = date_firsts_strs[1]
if date_lasts[0] < date_lasts[1]:
    date_last = date_lasts[1]
    date_last_str = date_lasts_strs[1]
elif date_lasts[0] >= date_lasts[1]:
    date_last = date_lasts[0]
    date_last_str = date_lasts_strs[0]

# Configure program
#SOMsizes = [[5, 5]]#[[2, 2], [3, 2], [3, 3], [4, 3], [4, 4], [5, 4], [5, 5]]
#for SOMsize in SOMsizes:
#SOMsize = [2, 2]#[4, 3]#[3, 2]  # size of the SOM grid
# print(SOMsize)
#SOMsize_str = str(SOMsize[0]) + 'x' + str(SOMsize[1])  # SOM grid size expressed as a string (e.g. '4x3')
# print(SOMsize_str)
mapSize = [175, 350]  # size of each of the future SOM maps
mapSize_1D = np.prod(mapSize)
# print(mapSize)
mapSize_str = str(mapSize[0]) + 'x' + str(mapSize[1])
# print(mapSize_str)

# Choose a region (NW, NC, NE, SW, SC, SE)
#myregion = input('Choose a region (NW, NC, NE, SW, SC, SE):\n')
myregion = str(sys.argv[1])
print('My region =', myregion)

if myregion == 'NW':
    maptedge = 1059#1  # top edge of the MRMS region
    mapbedge = 530#1750  # bottom "                      "
    mapledge = 1  # left "                      "
    mapredge = 900#3500  # right "                      "
elif myregion == 'NC':
    maptedge = 1059#1  # note that the top left corner is 1,1
    mapbedge = 530#1750
    mapledge = 451#1751
    mapredge = 1350#5250
elif myregion == 'NE':
    maptedge = 1059#1
    mapbedge = 530#1750
    mapledge = 900#3501
    mapredge = 1799#7000
elif myregion == 'SW':
    maptedge = 530#1751
    mapbedge = 1#1059#3500
    mapledge = 1
    mapredge = 900#3500
elif myregion == 'SC':
    maptedge = 530#1751
    mapbedge = 1#1059#3500
    mapledge = 451#1751
    mapredge = 1350#5250
elif myregion == 'SE':
    maptedge = 530#1751
    mapbedge = 1#1059#3500
    mapledge = 900#3501
    mapredge = 1799#7000
mapedges = [maptedge, mapbedge, mapledge, mapredge]

SOMsize_dim1 = int(sys.argv[2])
SOMsize_dim2 = int(sys.argv[3])
SOMsize = [SOMsize_dim1, SOMsize_dim2]
print('SOMsize =', SOMsize)

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_nums = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
# d = dict(month_nums, month_names)
d = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
     '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
# print(d['01'])
# print(int('01'))

print('Calling the mysom function (part 1) using MRMS data from', date_first, 'to', date_last, '...')
#[varSOM, dataArray, varSOM_nonzero, dataArray_nonzero, map_lats, map_lons, months_doubles, years_doubles,
#[varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles,
# year_month_strs, year_month_counts, mapedges] = mysom2(myfiles, varname, SOMsize, mapSize, myregion, d)
#[varSOMs, dataArray_train_allSOMs, dataArrays_allSOMs, map_lats, map_lons] = mysom2_multivar(myfiles, date_first, date_last, varnames, SOMsizes, mapSize, myregion, d)
#[varSOM, dataArray_train, dataArrays, percentile_99_allvars, indices_99th_allvars, count_99th_allvars, map_lats, map_lons] = mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapSize, myregion, d)
#[varSOM, dataArray_train, dataArrays, map_lats, map_lons] = mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapSize, myregion, d)
#[varSOM, dataArray_train, dataArrays, map_lats, map_lons] = mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d)
[dataArrays, map_lats, map_lons] = mysom2_multivar_thresh_part1(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d)
# print(np.prod(mapSize))
# print(varSOM.shape)
# print(varSOM)
# print(varSOM.get_weights())
# print(varSOM.get_weights().shape)

