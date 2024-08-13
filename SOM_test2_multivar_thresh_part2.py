# This script will create a SOM trained on MRMS MultiSensor 01H QPE

# Import necessary modules
print('Importing necessary modules')#, flush=True)
import glob
import sys
import numpy as np
from mysom2_multivar_thresh_part2 import mysom2_multivar_thresh_part2  # my own function for training the SOM on multiple variables, split in two parts to avoid time limit exceedence
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import xarray as xr
import datetime
import pickle
import math
from os.path import expanduser
import cartopy

# Point cartopy to offline data, because Hera compute nodes do not seem to have access to the internet
#cartopy.config['pre_existing_data_dir'] = expanduser('~/cartopy_data/')
#cartopy.config['pre_existing_data_dir'] = "/scratch1/BMC/wrfruc/naegele/cartopy_data/"
cartopy.config['pre_existing_data_dir'] = "/scratch1/BMC/wrfruc/naegele/mySOM2/lib/python3.12/site-packages/cartopy/data/shapefiles/natural_earth/physical/"
cartopy.config['pre_existing_data_dir'] = "/scratch1/BMC/wrfruc/naegele/mySOM2/lib/python3.12/site-packages/cartopy/data/shapefiles/natural_earth/cultural/"

ari = '1yr'
precip_duration1 = '1hr'
precip_duration2 = '24hr'

# Choose a region (NW, NC, NE, SW, SC, SE)
#myregion = input('Choose a region (NW, NC, NE, SW, SC, SE):\n')
myregion = str(sys.argv[1])
#myregion = 'SC'
print('My region =', myregion)#, flush=True)
SOMsize = [5, 5]
SOMsize_str = str(SOMsize[0]) + 'x' + str(SOMsize[1])  # SOM grid size expressed as a string (e.g. '4x3')
varname_pickle = 'radaronly_QPE_01H_24H'  # variable name for the pickle file where output from this script was saved
with open('/scratch2/BMC/wrfruc/naegele/SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' + myregion + '_vars_batch_part1_2021.pkl', 'rb') as f:  # Python 2: open(..., 'w')
#[varSOMs, dataArray, map_lats, map_lons] = mysom2_multivar(myfiles, varnames, SOMsizes, mapSize, myregion, d)
#            pickle.dump([varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles,
#                         year_month_strs, year_month_counts, mapedges, myfiles, varname, SOMsize, mapSize, myregion], f)
#        pickle.dump([varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars,
#                     year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapSize, myregion], f)
#        pickle.dump([varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars,
#                     year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapedges, myregion], f)
#        pickle.dump([dataArrays, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars,
#                     year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapedges, myregion], f)
        dataArrays, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars, year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapedges, myregion, shared_datetimes, datetimes_common = pickle.load(f)
        # To open the file later
        # with open('objs.pkl', 'rb') as f:  # Python 2: open(..., 'r') Python 2: open(..., 'rb')
        #     obj0, obj1, obj2, ... = pickle.load(f)

print('Reading in MRMS data ...')#, flush=True)
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
    print('len(myfiles[myvar]) = ' + str(len(myfiles[myvar])))#, flush=True)  # 
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
#SOMsize_dim1 = 5
#SOMsize_dim2 = 5
SOMsize = [SOMsize_dim1, SOMsize_dim2]
print('SOMsize =', SOMsize)#, flush=True)

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_nums = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
# d = dict(month_nums, month_names)
d = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
     '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
# print(d['01'])
# print(int('01'))


print('Calling the mysom function using MRMS data from', date_first, 'to', date_last, '...')#, flush=True)
#[varSOM, dataArray, varSOM_nonzero, dataArray_nonzero, map_lats, map_lons, months_doubles, years_doubles,
#[varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles,
# year_month_strs, year_month_counts, mapedges] = mysom2(myfiles, varname, SOMsize, mapSize, myregion, d)
#[varSOMs, dataArray_train_allSOMs, dataArrays_allSOMs, map_lats, map_lons] = mysom2_multivar(myfiles, date_first, date_last, varnames, SOMsizes, mapSize, myregion, d)
#[varSOM, dataArray_train, dataArrays, percentile_99_allvars, indices_99th_allvars, count_99th_allvars, map_lats, map_lons] = mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapSize, myregion, d)
#[varSOM, dataArray_train, dataArrays, map_lats, map_lons] = mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapSize, myregion, d)
#[varSOM, dataArray_train, dataArrays, map_lats, map_lons] = mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d)
[varSOM, dataArray_train] = mysom2_multivar_thresh_part2(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d, dataArrays, shared_datetimes, datetimes_common)
#[varSOM, predictions, dataArray_train] = mysom2_multivar_thresh_part2(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d, dataArrays, shared_datetimes, datetimes_common)
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
winning_nodes = varSOM.win_map(dataArray_train, return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
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

for myvar in range(0, len(mypaths)):
    print('Creating ' + SOMsize_str + ' SOM figures for ' + varname_titles[myvar] + ' ' + myregion + '...')
    fig = plt.figure(figsize=(12, 6))
    node_count = -1
    my_var_cmap = 'Blues'
    my_var_vmin = 0
    if myvar == 0:
        my_var_vmax = 1#5#25#5
        my_var_levels = [0,0.2,0.4,0.6,0.8,1]#[0,1,2,3,4,5]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
        my_var_tick_levels = [0,0.2,0.4,0.6,0.8,1]#[0,1,2,3,4,5]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
    elif myvar == 1:
        my_var_vmax = 20#10#25#5
        my_var_levels = [0,4,8,12,16,20]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
        my_var_tick_levels = [0,4,8,12,16,20]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
    my_var_cbarextend = 'max'
    my_var_cbarlabel = 'liquid precipitation [mm]'
    for iSOM in range(0, SOMsize[1]):
        for jSOM in range(0, SOMsize[0]):
            node_count = node_count + 1

            winning_node_indices = winning_nodes[jSOM, iSOM]
#            winning_node_indices = np.where(predictions == node_count)

            print(str(SOMsize) + ': jSOM (SOMsize[0]) = ' + str(jSOM) + ', iSOM (SOMsize[1]) = ' + str(iSOM))
#                print('SOM_counter = ' + str(SOM_counter))
            print('myvar = ' + str(myvar))
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
            print('dataArrays[myvar].shape = ' + str(dataArrays[myvar].shape))
            dataArray_node = dataArrays[myvar][winning_node_indices]
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
            print('dataArray_node.shape = ' + str(dataArray_node.shape))  # 86, 61250 in example
#            print('np.nanmin(dataArray_node) = ' + str(np.nanmin(dataArray_node)))  # 
#            print('np.nanmax(dataArray_node) = ' + str(np.nanmax(dataArray_node)))  # 
#            dataArray_node_map = np.nanmean(dataArray_node.reshape(-1, mapSize[1], mapSize[0]), axis=0)  # -1 --> value is inferred from the length of the array and remaining dimensions (keeps 1st dim len consistent if 2nd dim len = mapSize[1] * mapSize[0])
#            dataArray_node_map = np.nanmean(dataArray_node.reshape(-1, math.ceil((mapredge-mapledge)/10), math.ceil((mapbedge-maptedge)/10)), axis=0)  # -1 --> value is inferred from the length of the array and remaining dimensions (keeps 1st dim len consistent if 2nd dim len = mapSize[1] * mapSize[0])
            dataArray_node_map = np.nanmean(dataArray_node.reshape(-1, math.ceil((mapredge-mapledge)/10), math.ceil((maptedge-mapbedge)/10)), axis=0)  # -1 --> value is inferred from the length of the array and remaining dimensions (keeps 1st dim len consistent if 2nd dim len = mapSize[1] * mapSize[0])
#            dataArray_node_map_dummy = np.full((len(dataArray_node_map), len(dataArray_node_map[0])), 0.5)
            # ValueError: cannot reshape array of size 22752900 into shape (89,52)
#                print('Printing dataArray_node_map for ' + SOMsize_str + ' node ' + str(node_count))
#                print(dataArray_node_map)
            print('dataArray_node_map.shape = ' + str(dataArray_node_map.shape))
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
    varnames = ['mrms_radaronly_01h', 'mrms_radaronly_24h']
    plt.suptitle(varnames[myvar] + ' -- ' + date_first_str + '-' + date_last_str, x=0.5125, y=0.9125, fontsize=14)
    plt.savefig('/scratch2/BMC/wrfruc/naegele/SOM_01H24Htrain_' + varnames[myvar] + '_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + date_first_str + '_' + date_last_str + '_' + myregion + '_test_zeros.png')#, bbox_inches='tight', pad_inches=0)


