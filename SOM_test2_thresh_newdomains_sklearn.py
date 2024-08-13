# This script will create a SOM trained on MRMS MultiSensor 01H QPE

# Import necessary modules
import glob
import sys
import numpy as np
#from mysom2_multivar_thresh_newdomains_sklearn import mysom2_multivar_thresh_newdomains_sklearn  # my own function for training the SOM on multiple variables, split in two to avoid time limit exceedence
from mysom2_thresh_newdomains_sklearn import mysom2_thresh_newdomains_sklearn  # my own function for training the SOM on just one variable, split in two to avoid time limit exceedence
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import xarray as xr
import datetime
import pickle
import math
from os.path import expanduser
import cartopy

ari = '1yr'
precip_duration1 = '1hr'
precip_duration2 = '24hr'
#precip_durations = [precip_duration1]#, precip_duration2]
precip_duration = str(sys.argv[1])

print('Reading in MRMS data ...')
#mypath = '/Users/steven.naegele/Data/MRMS/20210101_20231231/'
#mypath = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_01H_Pass2/20210101_20231231/'
#mypath1 = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_01H_Pass2/20210101_20231231/'
#mypath2 = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_24H_Pass2/20210101_20231231/'
mypath1 = '/scratch1/BMC/wrfruc/ejames/climo/mrms/01h/**/'
mypath2 = '/scratch1/BMC/wrfruc/ejames/climo/mrms/24h/**/'
mypaths = [mypath1]#, mypath2
varnames = ['mrms_radaronly_01h', 'mrms_radaronly_24h']
if precip_duration == precip_duration1:
    mypath = mypath1
    output_interval = 6
    varname = varnames[0]
elif precip_duration == precip_duration2:
    mypath = mypath2
    output_interval = 24
    varname = varnames[1]
#myfilenames1 = 'MultiSensor_QPE_01H_Pass2_00.00_202*0000.nc'
#myfilenames2 = 'MultiSensor_QPE_24H_Pass2_00.00_202*0000.nc'
#myfilenames = [myfilenames1, myfilenames2]
myfilenames = 'mrms_radaronly_20*.nc'
myfiles = []
nMaps = []
filename_firsts = []
#varnames = []
varname_titles = []
date_firsts = []
date_firsts_strs = []
filename_lasts = []
date_lasts = []
date_lasts_strs = []
# for myfile in sorted(glob.glob(mypath + 'MultiSensor_QPE_01H_Pass2_00.00_2021022*.grib2')):
#for myfile in sorted(glob.glob(mypath + 'MultiSensor_QPE_01H_Pass2_00.00_20*0000.nc')):
#for myvar in range(0, 1):#len(mypaths)):
#    myfiles.append(sorted(glob.glob(mypaths[myvar] + myfilenames[myvar])))
#    myfiles.append(sorted(glob.glob(mypaths[myvar] + myfilenames)))
myfiles.append(sorted(glob.glob(mypath + myfilenames)))
#myfiles2 = []
#for myfile2 in sorted(glob.glob(mypath2 + 'MultiSensor_QPE_24H_Pass2_00.00_20*0000.nc')):
#    myfiles2.append(myfile2)

#    nMaps.append(len(myfiles[myvar]))  # 26240, 26235
nMaps.append(len(myfiles))  # 26240, 26235
#    print('len(myfiles[myvar]) = ' + str(len(myfiles[myvar])))  # 
print('len(myfiles) = ' + str(len(myfiles)))  # 
#print(nMaps1)  # 26240
#print(nMaps2)  # 26235
#    example_full_filename = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_01H_Pass2/20210101_20231231/MultiSensor_QPE_01H_Pass2_00.00_20210101-000000.nc'
example_full_filename = '/scratch1/BMC/wrfruc/ejames/climo/mrms/01h/202212/mrms_radaronly_2022123123.nc'

#    sample_filename = 'MultiSensor_QPE_01H_Pass2_00.00_20210101-000000.nc'
#    filename_first = myfiles[myvar][0]
filename_first = myfiles[0][0]
filename_firsts.append(filename_first)
#    varname = filename_first.split('/')[-1].split('.')[0][0:-3]
#varname = "_".join(filename_first.split("_", 2)[:2])  # split the filename up to the second occurrence of "_" and combine the first two elements
#varnames.append(varname)
varname_title = varname.replace('_',' ')
varname_titles.append(varname_title)
#    date_first = filename_first.split('/')[-1].split('.')[1][3:10+1]
#    date_first = filename_first.split('_')[-1][0:7+1]
date_first = filename_first.split('_')[-1][0:9+1]
date_firsts_strs.append(date_first)
#    date_first = datetime.datetime.strptime(date_first, '%Y%m%d')
date_first = datetime.datetime.strptime(date_first, '%Y%m%d%H')
date_first_alt = filename_first.split('_')[-1][0:7+1] + '12'#'00'
date_first_alt = datetime.datetime.strptime(date_first_alt, '%Y%m%d%H')
while date_first > date_first_alt:
#        date_first_alt = date_first_alt + datetime.timedelta(hours=6)
    date_first_alt = date_first_alt + datetime.timedelta(hours=output_interval)
#    date_firsts.append(date_first)
date_firsts.append(date_first_alt)
#    filename_last = myfiles[myvar][-1]
filename_last = myfiles[0][-1]
filename_lasts.append(filename_last)
#    date_last = filename_last.split('/')[-1].split('.')[1][3:10+1]
#    date_last = filename_last.split('_')[-1][0:7+1]
date_last = filename_last.split('_')[-1][0:9+1]
date_lasts_strs.append(date_last)
#    date_last = datetime.datetime.strptime(date_last, '%Y%m%d')
date_last = datetime.datetime.strptime(date_last, '%Y%m%d%H')
date_last_alt = filename_last.split('_')[-1][0:7+1] + '12'#'18'
date_last_alt = datetime.datetime.strptime(date_last_alt, '%Y%m%d%H')
while date_last_alt > date_last:
#        date_last_alt = date_last_alt - datetime.timedelta(hours=6)
    date_last_alt = date_last_alt - datetime.timedelta(hours=output_interval)
#    date_lasts.append(date_last)
date_lasts.append(date_last_alt)
#filename_first2 = myfiles2[0]
#varname2 = filename_first2.split('/')[-1].split('.')[0][0:-3]
#varname_title2 = varname2.replace('_',' ')
#date_first2 = filename_first2.split('/')[-1].split('.')[1][3:10+1]
#date_first2 = datetime.datetime.strptime(date_first2, '%Y%m%d')
#filename_last2 = myfiles2[-1]
#date_last2 = filename_last2.split('/')[-1].split('.')[1][3:10+1]
#date_last2 = datetime.datetime.strptime(date_last2, '%Y%m%d')
#if date_firsts[0] < date_firsts[1]:
#    date_first = date_firsts[0]
#    date_first_str = date_firsts_strs[0]
#elif date_firsts[0] >= date_firsts[1]:
#    date_first = date_firsts[1]
#    date_first_str = date_firsts_strs[1]
#if date_lasts[0] < date_lasts[1]:
#    date_last = date_lasts[1]
#    date_last_str = date_lasts_strs[1]
#elif date_lasts[0] >= date_lasts[1]:
#    date_last = date_lasts[0]
#    date_last_str = date_lasts_strs[0]

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
myregion = str(sys.argv[2])
print('My region =', myregion)

if myregion == 'NW':
    maptedge = 1059#1  # top edge of the MRMS region
    mapbedge = 530#1750  # bottom "                      "
    mapledge = 1  # left "                      "
    mapredge = 900#3500  # right "                      "
elif myregion == 'NC':
    maptedge = 850#1059#1  # note that the top left corner is 1,1
    mapbedge = 500#530#1750
    mapledge = 551#451#1751
    mapredge = 1300#1350#5250
elif myregion == 'NE':
    maptedge = 850#1059#1
    mapbedge = 701#530#1750
    mapledge = 1426#900#3501
    mapredge = 1650#1799#7000
elif myregion == 'SW':
    maptedge = 650#530#1751
    mapbedge = 351#1#1059#3500
    mapledge = 501#1
    mapredge = 750#900#3500
elif myregion == 'SC':
    maptedge = 530#550#500#530#1751
    mapbedge = 1#1059#3500
    mapledge = 551#451#1751
    mapredge = 1350#1400#1050#1500#1350#5250
elif myregion == 'SE':
    maptedge = 530#1751
    mapbedge = 1#1059#3500
    mapledge = 900#3501
    mapredge = 1799#7000
elif myregion == 'MA':  # Mid-Atlantic
    maptedge = 700
    mapbedge = 451
    mapledge = 1301
    mapredge = 1550
mapedges = [maptedge, mapbedge, mapledge, mapredge]

SOMsize_dim1 = int(sys.argv[3])
SOMsize_dim2 = int(sys.argv[4])
SOMsize = [SOMsize_dim1, SOMsize_dim2]
print('SOMsize =', SOMsize)
SOMsize_str = str(SOMsize[0]) + 'x' + str(SOMsize[1])  # SOM grid size expressed as a string (e.g. '4x3')

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_nums = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
# d = dict(month_nums, month_names)
d = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
     '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
# print(d['01'])
# print(int('01'))

#for myvar in range(0, len(mypaths)):
#print('Calling the mysom function using MRMS data from', date_first, 'to', date_last, '...')
#print('Calling the mysom function for ' + varnames[myvar] + ' using MRMS data from', date_firsts[myvar], 'to', date_lasts[myvar], '...')
print('Calling the mysom function for ' + varname + ' using MRMS data from', date_first_alt, 'to', date_last_alt, '...')
#[varSOM, dataArray, varSOM_nonzero, dataArray_nonzero, map_lats, map_lons, months_doubles, years_doubles,
#[varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles,
# year_month_strs, year_month_counts, mapedges] = mysom2(myfiles, varname, SOMsize, mapSize, myregion, d)
#[varSOMs, dataArray_train_allSOMs, dataArrays_allSOMs, map_lats, map_lons] = mysom2_multivar(myfiles, date_first, date_last, varnames, SOMsizes, mapSize, myregion, d)
#[varSOM, dataArray_train, dataArrays, percentile_99_allvars, indices_99th_allvars, count_99th_allvars, map_lats, map_lons] = mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapSize, myregion, d)
#[varSOM, dataArray_train, dataArrays, map_lats, map_lons] = mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapSize, myregion, d)
#[varSOM, dataArray_train, dataArrays, map_lats, map_lons] = mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d)
#[dataArrays, map_lats, map_lons] = mysom2_multivar_thresh_newdomains(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d)
#[varSOM, dataArrays, map_lats, map_lons, dataArray_train] = mysom2_multivar_thresh_newdomains(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d)
#[dataArrays, map_lats, map_lons, dataArray_train] = mysom2_multivar_thresh_newdomains(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d)
#[varSOM, predictions, dataArrays, map_lats, map_lons, dataArray_train] = mysom2_multivar_thresh_newdomains_sklearn(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d)
#    [varSOM, predictions, dataArray, map_lats, map_lons, dataArray_train] = mysom2_thresh_newdomains_sklearn(myfiles[myvar], date_firsts[myvar], date_lasts[myvar], varnames[myvar], SOMsize, mapedges, myregion, d)
[varSOM, predictions, dataArray, map_lats, map_lons, dataArray_train] = mysom2_thresh_newdomains_sklearn(myfiles, date_first_alt, date_last_alt, varname, SOMsize, mapedges, myregion, d, precip_duration)
# print(np.prod(mapSize))
# print(varSOM.shape)
# print(varSOM)
# print(varSOM.get_weights())
# print(varSOM.get_weights().shape)

#np.set_printoptions(threshold=sys.maxsize)
print('varSOM = ' + str(varSOM))  #
#print('dataArray_train = ' + str(dataArray_train))  #
#print(dataArray_train)
#np.set_printoptions(threshold=False)

#SOM_counter = -1
#for SOMsize in SOMsizes:
#    print(SOMsize)
#    SOM_counter = SOM_counter + 1
#SOMsize_str = str(SOMsize[0]) + 'x' + str(SOMsize[1])  # SOM grid size expressed as a string (e.g. '4x3')
#
#    varSOM = varSOMs[SOM_counter]
#
#    print('len(dataArray_train_allSOMs) = ' + str(len(dataArray_train_allSOMs[SOM_counter])))
##    winning_nodes = varSOM.win_map(dataArray_train_allSOMs[SOM_counter], return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
#winning_nodes = varSOM.win_map(dataArray_train, return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
##    winning_nodes = varSOM.win_map(dataArray_train_allSOMs[0], return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
                                                                                     # note that the SOM node indices are [col, row], and I will read through cols first
#print('winning_nodes:')
#print(winning_nodes)

## Find the common elements in two lists
#def common_member(a, b):
#    a_set = set(a)
#    b_set = set(b)
#
#    if (a_set & b_set):
#        common_members = sorted(a_set & b_set)  # convert set to a list
#    else:
#        common_members = []
#
#    return common_members

#for myvar in range(0, len(mypaths)):
#    print('Creating ' + SOMsize_str + ' SOM figures for ' + varname_titles[myvar] + ' ' + myregion + '...')
print('Creating ' + SOMsize_str + ' SOM figures for ' + varname_title + ' ' + myregion + '...')
fig = plt.figure(figsize=(12, 6))
node_count = -1
my_var_cmap = 'Blues'
my_var_vmin = 0
#    if myvar == 0:
if precip_duration == precip_duration1:
    my_var_vmax = 0.5#1#5#25#5
    my_var_levels = [0,0.1,0.2,0.3,0.4,0.5]#[0,0.2,0.4,0.6,0.8,1]#[0,1,2,3,4,5]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
    my_var_tick_levels = [0,0.1,0.2,0.3,0.4,0.5]#[0,0.2,0.4,0.6,0.8,1]#[0,1,2,3,4,5]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
#    elif myvar == 1:
elif precip_duration == precip_duration2:
    my_var_vmax = 10#20#10#25#5
    my_var_levels = [0,2,4,6,8,10]#[0,4,8,12,16,20]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
    my_var_tick_levels = [0,2,4,6,8,10]#[0,4,8,12,16,20]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
my_var_cbarextend = 'max'
my_var_cbarlabel = 'liquid precipitation [mm]'
for iSOM in range(0, SOMsize[1]):
    for jSOM in range(0, SOMsize[0]):
        node_count = node_count + 1

#            winning_node_indices = winning_nodes[jSOM, iSOM]
        winning_node_indices = np.where(predictions == node_count)[0]

        print(str(SOMsize) + ': jSOM (SOMsize[0]) = ' + str(jSOM) + ', iSOM (SOMsize[1]) = ' + str(iSOM))
#                print('SOM_counter = ' + str(SOM_counter))
#            print('myvar = ' + str(myvar))
        print('winning_node_indices:')
        print(winning_node_indices)
#            print('len(winning_node_indices) = ' + str(len(winning_node_indices)))
#                if myvar == 0:
#                print('dataArrays_allSOMs:')
#                print(dataArrays_allSOMs)
#                print('NaN count for dataArrays_allSOMs = ' + str(np.count_nonzero(np.isnan(dataArrays_allSOMs))))
#                print(dataArrays_allSOMs[0])
#                print('NaN count for dataArrays_allSOMs[0] = ' + str(np.count_nonzero(np.isnan(dataArrays_allSOMs[0]))))
#                print(dataArrays_allSOMs[0][0])
#                print('NaN count for dataArrays_allSOMs[0][0] = ' + str(np.count_nonzero(np.isnan(dataArrays_allSOMs[0][0]))))
#                print(dataArrays_allSOMs[0][0][0])
#                print('NaN count for dataArrays_allSOMs[0][0][0] = ' + str(np.count_nonzero(np.isnan(dataArrays_allSOMs[0][0][0]))))
#            print('dataArrays_allSOMs dimensions:')
#            print(len(dataArrays_allSOMs))  # 7 (w/ allSOMs)
#            print(len(dataArrays_allSOMs[0]))  # 2 (w/ allSOMs)
#            print(len(dataArrays_allSOMs[0][0]))  # 215 (w/ allSOMs)
#                print(len(dataArrays_allSOMs[0][0][0]))  # 61250 (w/ allSOMs)
                    #dataArray_node = dataArrays_allSOMs[SOM_counter][myvar][0:mapSize_1D, winning_node_indices]
#                    dataArray_node = dataArrays_allSOMs[SOM_counter][myvar][winning_node_indices, 0:mapSize_1D]
#                dataArray_node = dataArrays_allSOMs[SOM_counter][myvar][winning_node_indices]
#            winning_node_thresh_indices = common_members(winning_node_indices, indices_99th_allvars[myvar])

#            print('dataArrays[myvar].shape = ' + str(dataArrays[myvar].shape))
#            dataArray_node = dataArrays[myvar][winning_node_indices]
#        print('dataArray.shape = ' + str(dataArray.shape))  # now a list, so no shape (?) 07 Aug 2024
        dataArray_node = np.array(dataArray)[winning_node_indices]

#            print('min(dataArray_node) = ' + str(np.min(dataArray_node)))  # see if there are any NaNs, because some nodes aren't being plotted when using a threshold
#            print('max(dataArray_node) = ' + str(np.max(dataArray_node)))
#                elif myvar == 1:
#                    print('dataArrays_allSOMs dimensions:')
#                    print(len(dataArrays_allSOMs))  # 7
#                    print(len(dataArrays_allSOMs[0]))  # 2
#                    print(len(dataArrays_allSOMs[0][0]))  # 215
#                    print(len(dataArrays_allSOMs[0][0][0]))  # 61250
#                    #dataArray_node = dataArrays_allSOMs[SOM_counter][myvar][mapSize_1D+1:2*mapSize_1D, winning_node_indices]
##                    dataArray_node = dataArrays_allSOMs[SOM_counter][myvar][winning_node_indices, mapSize_1D+1:2*mapSize_1D]
#                    dataArray_node = dataArrays_allSOMs[SOM_counter][myvar][winning_node_indices]
#                print('Printing dataArray_node')
#                print(dataArray_node)
#        print('dataArray_node.shape = ' + str(dataArray_node.shape))  # 86, 61250 in example
#            print('np.nanmin(dataArray_node) = ' + str(np.nanmin(dataArray_node)))  # 
#            print('np.nanmax(dataArray_node) = ' + str(np.nanmax(dataArray_node)))  # 
#            dataArray_node_map = np.nanmean(dataArray_node.reshape(-1, mapSize[1], mapSize[0]), axis=0)  # -1 --> value is inferred from the length of the array and remaining dimensions (keeps 1st dim len consistent if 2nd dim len = mapSize[1] * mapSize[0])
#            dataArray_node_map = np.nanmean(dataArray_node.reshape(-1, math.ceil((mapredge-mapledge)/10), math.ceil((mapbedge-maptedge)/10)), axis=0)  # -1 --> value is inferred from the length of the array and remaining dimensions (keeps 1st dim len consistent if 2nd dim len = mapSize[1] * mapSize[0])
        dataArray_node_map = np.nanmean(dataArray_node.reshape(-1, math.ceil((mapredge-mapledge)/10), math.ceil((maptedge-mapbedge)/10)), axis=0)  # -1 --> value is inferred from the length of the array and remaining dimensions (keeps 1st dim len consistent if 2nd dim len = mapSize[1] * mapSize[0])
#            dataArray_node_map_dummy = np.full((len(dataArray_node_map), len(dataArray_node_map[0])), 0.5)
            # ValueError: cannot reshape array of size 22752900 into shape (89,52)
#                print('Printing dataArray_node_map for ' + SOMsize_str + ' node ' + str(node_count))
#                print(dataArray_node_map)
#        print('dataArray_node_map.shape = ' + str(dataArray_node_map.shape))
#            dataArray_node_map = [[x if not np.isnan(x) else x == 0 for x in y] for y in dataArray_node_map]
        print('dataArray_node_map = ' + str(dataArray_node_map))
            # Adjust the number and arrangement of the subplots based on the SOM grid size
        if SOMsize_str == '3x2':
            my_subplot = 231
        elif SOMsize_str == '4x2':
            my_subplot = 241
        elif SOMsize_str == '4x3':
            my_subplot = 341
        elif SOMsize_str == '5x4':
            my_subplot = 451
        elif SOMsize_str == '2x2':
            my_subplot = 221
        elif SOMsize_str == '3x3':
            my_subplot = 331
        elif SOMsize_str == '4x4':
            my_subplot = 441
        elif SOMsize_str == '5x5':
            my_subplot = 551
                # cart_proj = ccrs.LambertConformal()
        cart_proj = ccrs.PlateCarree()
        ax = plt.subplot(SOMsize[1], SOMsize[0], node_count + 1, projection=cart_proj)
        print('map_lons = ' + str(map_lons))
        print('map_lats = ' + str(map_lats))
#                ax = plt.subplot(SOMsize[1], SOMsize[0], node_count, projection=cart_proj)
#                varSOM_plot = varSOM.get_weights()[jSOM-1, iSOM-1, :].reshape(mapSize[1], mapSize[0])
#                print(varSOM.get_weights()[jSOM-1, iSOM-1, :])
#                print('NaN count for varSOM weights at current node = ' + str(np.count_nonzero(np.isnan(varSOM.get_weights()[jSOM-1, iSOM-1, :]))))
#                ax_ = ax.contourf(map_lons, map_lats, np.transpose(varSOM_plot), cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
        ax_ = ax.contourf(map_lons, map_lats, np.transpose(dataArray_node_map), cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
                                          levels=my_var_levels, transform=ccrs.PlateCarree(), extend=my_var_cbarextend)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=1.00, linestyle='--')
        gl.xlabels_top = False
        gl.xlabel_style = {'size': 6, 'color': 'black'}
        gl.ylabels_left = False
        gl.ylabel_style = {'size': 6, 'color': 'black'}
        ax.coastlines('10m', linewidth=0.8)
        country_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land',
                                                       scale='10m', facecolor='none')
        state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
                                                     scale='10m', facecolor='none')
        ax.add_feature(country_borders, edgecolor='black')
        ax.add_feature(state_borders, edgecolor='black', linewidth=0.5)
        ax.title.set_text('Node ' + str(node_count + 1))
                # ax.outline_patch.set_alpha(0)  # plot outline/gridbox
                # ax.background_patch.set_alpha(0)  # plot background
                # plt.imshow(varSOM_plot, interpolation='none', aspect='auto')
cax = plt.axes([0.2, 0.525, 0.625, 0.025])
cbar = fig.colorbar(ax_, cax=cax, orientation='horizontal', extend=my_var_cbarextend,
                    ticks=my_var_tick_levels, label=my_var_cbarlabel)#, pad=0.05)
#    plt.suptitle(varnames[myvar] + ' -- ' + date_first_strs[myvar] + '-' + date_last_strs[myvar], x=0.5125, y=0.9125, fontsize=14)
#plt.suptitle(varname + ' -- ' + date_first_str + '-' + date_last_str, x=0.5125, y=0.9125, fontsize=14)
plt.suptitle(varname + ' -- ' + date_firsts_strs[0] + '-' + date_lasts_strs[0], x=0.5125, y=0.9125, fontsize=14)
#    plt.savefig('/scratch2/BMC/wrfruc/naegele/SOM_01H24Htrain_' + varnames[myvar] + '_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + date_first_str + '_' + date_last_str + '_' + myregion + '_test_newdomains_small_sklearn.png')#, bbox_inches='tight', pad_inches=0)
#    plt.savefig('/scratch2/BMC/wrfruc/naegele/SOM_01H24Htrain_' + varnames[myvar] + '_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + date_first_str + '_' + date_last_str + '_' + myregion + '_test_newdomains_sklearn.png')#, bbox_inches='tight', pad_inches=0)

#    plt.savefig('/scratch2/BMC/wrfruc/naegele/SOM_01H24Htrain_' + varnames[myvar] + '_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + date_first_str + '_' + date_last_str + '_' + myregion + '_test_newdomains_sklearn_cbar.png')#, bbox_inches='tight', pad_inches=0)
#    plt.savefig('/scratch2/BMC/wrfruc/naegele/SOM_' + varnames[myvar][-3:-1] + 'Htrain_' + varnames[myvar] + '_' + SOMsize_str + '_thresh_ari' + ari + precip_duration + '_' + date_first_strs[myvar] + '_' + date_last_strs[myvar] + '_' + myregion + '_test_newdomains_sklearn_cbar_6hr_test.png')#, bbox_inches='tight', pad_inches=0)
#plt.savefig('/scratch2/BMC/wrfruc/naegele/SOM_' + varname[-3:-1] + 'Htrain_' + varname + '_' + SOMsize_str + '_thresh_ari' + ari + precip_duration + '_' + date_first_str + '_' + date_last_str + '_' + myregion + '_test_newdomains_sklearn_cbar_6hr_test.png')#, bbox_inches='tight', pad_inches=0)
plt.savefig('/scratch2/BMC/wrfruc/naegele/SOM_' + varname[-3:-1] + 'Htrain_' + varname + '_' + SOMsize_str + '_thresh_ari' + ari + precip_duration + '_' + date_firsts_strs[0] + '_' + date_lasts_strs[0] + '_' + myregion + '_test_newdomains_sklearn_cbar_' + str(output_interval) + 'hr_test.png')#, bbox_inches='tight', pad_inches=0)


