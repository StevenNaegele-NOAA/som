# This script will create heatmaps of the probability a SOM node will persist or transition to another node after a given period of hours

# Import necessary modules
import glob
import sys
import numpy as np
from mysom2_multivar import mysom2_multivar  # my own function for training the SOM on multiple variables
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import xarray as xr
import datetime
import pickle
import seaborn as sns

print('Reading in MRMS data ...')
##mypath = '/Users/steven.naegele/Data/MRMS/20210101_20231231/'
##mypath = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_01H_Pass2/20210101_20231231/'
#mypath1 = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_01H_Pass2/20210101_20231231/'
#mypath2 = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_24H_Pass2/20210101_20231231/'
#mypaths = [mypath1, mypath2]
#myfilenames1 = 'MultiSensor_QPE_01H_Pass2_00.00_202*0000.nc'
#myfilenames2 = 'MultiSensor_QPE_24H_Pass2_00.00_202*0000.nc'
#myfilenames = [myfilenames1, myfilenames2]
#myfiles = []
#nMaps = []
#filename_firsts = []
#varnames = []
#varname_titles = []
date_firsts = []
date_firsts_strs = []
#filename_lasts = []
date_lasts = []
date_lasts_strs = []
## for myfile in sorted(glob.glob(mypath + 'MultiSensor_QPE_01H_Pass2_00.00_2021022*.grib2')):
##for myfile in sorted(glob.glob(mypath + 'MultiSensor_QPE_01H_Pass2_00.00_20*0000.nc')):

# Choose a region (NW, NC, NE, SW, SC, SE)
myregion_plot = str(sys.argv[1])
print('My region =', myregion_plot)

SOMsize_dim1 = int(sys.argv[2])
SOMsize_dim2 = int(sys.argv[3])
SOMsize = [SOMsize_dim1, SOMsize_dim2]
print('SOMsize =', SOMsize)
SOMsize_str = str(SOMsize[0]) + 'x' + str(SOMsize[1])  # SOM grid size expressed as a string (e.g. '4x3')

ari = '1yr'
precip_duration1 = '1hr'
precip_duration2 = '24hr'
varname_pickle = 'radaronly_QPE_01H_24H'  # variable name for the pickle file where output from this script will be saved
#with open('SOM_' + SOMsize_str + '_learningrate25_MultiSensor_QPE_01H_24H_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
#with open('SOM_' + SOMsize_str + '_thresh_percentile_99th_MultiSensor_QPE_01H_24H_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
with open('SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' +  myregion_plot + '_vars_batch_part2.pkl', 'rb') as f:
    varSOM, dataArrays_orig, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars, \
    year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapedges, myregion = pickle.load(f)

for myvar in range(0, len(varnames)):
#    myfiles.append(sorted(glob.glob(mypaths[myvar] + myfilenames[myvar])))
#myfiles2 = []
#for myfile2 in sorted(glob.glob(mypath2 + 'MultiSensor_QPE_24H_Pass2_00.00_20*0000.nc')):
#    myfiles2.append(myfile2)

#    nMaps.append(len(myfiles[myvar]))  # 26240, 26235

#    filename = myfiles[myvar][0]
#    hour = filename.split('/')[-1].split('.')[1][12:13+1]

#print(nMaps1)  # 26240
#print(nMaps2)  # 26235
#    example_full_filename = '/work2/noaa/wrfruc/snaegele/MRMS_MultiSensor_QPE_01H_Pass2/20210101_20231231/MultiSensor_QPE_01H_Pass2_00.00_20210101-000000.nc'
#
#    sample_filename = 'MultiSensor_QPE_01H_Pass2_00.00_20210101-000000.nc'
    filename_first = myfiles[myvar][0]
#    filename_firsts.append(filename_first)
#    varname = filename_first.split('/')[-1].split('.')[0][0:-3]
#    varnames.append(varname)
#    varname_title = varname.replace('_',' ')
#    varname_titles.append(varname_title)
    date_first = filename_first.split('/')[-1].split('.')[1][3:10+1]
    date_firsts_strs.append(date_first)
    date_first = datetime.datetime.strptime(date_first, '%Y%m%d')
    date_firsts.append(date_first)
    filename_last = myfiles[myvar][-1]
#    filename_lasts.append(filename_last)
    date_last = filename_last.split('/')[-1].split('.')[1][3:10+1]
    date_lasts_strs.append(date_last)
    date_last = datetime.datetime.strptime(date_last, '%Y%m%d')
    date_lasts.append(date_last)
##filename_first2 = myfiles2[0]
##varname2 = filename_first2.split('/')[-1].split('.')[0][0:-3]
##varname_title2 = varname2.replace('_',' ')
##date_first2 = filename_first2.split('/')[-1].split('.')[1][3:10+1]
##date_first2 = datetime.datetime.strptime(date_first2, '%Y%m%d')
##filename_last2 = myfiles2[-1]
##date_last2 = filename_last2.split('/')[-1].split('.')[1][3:10+1]
##date_last2 = datetime.datetime.strptime(date_last2, '%Y%m%d')
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
#mapSize = [175, 350]  # size of each of the future SOM maps
#mapSize_1D = np.prod(mapSize)
## print(mapSize)
#mapSize_str = str(mapSize[0]) + 'x' + str(mapSize[1])
## print(mapSize_str)

## Choose a region (NW, NC, NE, SW, SC, SE)
##myregion = input('Choose a region (NW, NC, NE, SW, SC, SE):\n')
#myregion = str(sys.argv[1])
#print('My region =', myregion)
#
#SOMsize_dim1 = int(sys.argv[2])
#SOMsize_dim2 = int(sys.argv[3])
#SOMsize = [SOMsize_dim1, SOMsize_dim2]
#print('SOMsize =', SOMsize)

#month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
#month_nums = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
## d = dict(month_nums, month_names)
#d = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
#     '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
# print(d['01'])
# print(int('01'))

#print('Calling the mysom function using MRMS data from', date_first, 'to', date_last, '...')
#[varSOM, dataArray, varSOM_nonzero, dataArray_nonzero, map_lats, map_lons, months_doubles, years_doubles,
#[varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles,
# year_month_strs, year_month_counts, mapedges] = mysom2(myfiles, varname, SOMsize, mapSize, myregion, d)
#[varSOMs, dataArray_train_allSOMs, dataArrays_allSOMs, map_lats, map_lons] = mysom2_multivar(myfiles, date_first, date_last, varnames, SOMsizes, mapSize, myregion, d)
#[varSOM, dataArray_train, dataArrays, map_lats, map_lons] = mysom2_multivar(myfiles, date_first, date_last, varnames, SOMsize, mapSize, myregion, d)
# print(np.prod(mapSize))
# print(varSOM.shape)
# print(varSOM)
# print(varSOM.get_weights())
# print(varSOM.get_weights().shape)

#SOM_counter = -1
#for SOMsize in SOMsizes:
#    print(SOMsize)
#    SOM_counter = SOM_counter + 1
#SOMsize_str = str(SOMsize[0]) + 'x' + str(SOMsize[1])  # SOM grid size expressed as a string (e.g. '4x3')
#
#    varSOM = varSOMs[SOM_counter]
#
#    print('len(dataArray_train_allSOMs) = ' + str(len(dataArray_train_allSOMs[SOM_counter])))
#    winning_nodes = varSOM.win_map(dataArray_train_allSOMs[SOM_counter], return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
winning_nodes = varSOM.win_map(dataArray_train, return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
#    winning_nodes = varSOM.win_map(dataArray_train_allSOMs[0], return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
                                                                                     # note that the SOM node indices are [col, row], and I will read through cols first

print('len(winning_nodes) = ' + str(len(winning_nodes)))
print('winning_nodes = ' + str(winning_nodes))

hours = [1,2,3,6,9,12,18,24,36,48,72,96,120,144,168,192,216,240]  # how many hours between node transitions

winning_node_numbers_with_time = np.zeros((len(dataArray_train), 1))
for i in range(0, len(dataArray_train)):
#    winning_node_2d_grid_dim = list(winning_nodes.keys())[list(winning_nodes.values()).index(i)]
#    winning_node_2d_grid_dim = np.where(np.isin(list(myindices.values())[i], 70))
    for j in range(0, len(winning_nodes.values())):
        if any(np.isin(list(winning_nodes.values())[j], i)):
            # check if the current index is in each dict value for the 2d SOM node and get 2d grid node dim (e.g. (4, 2))
            print('list(winning_nodes.keys()) = ' + str(list(winning_nodes.keys())))
            print('len(winning_nodes.values()) = ' + str(len(winning_nodes.values())))
            print('list(winning_nodes.values())[j] = ' + str(list(winning_nodes.values())[j]))
            print('np.isin(list(winning_nodes.values())[j], i) = ' + str(np.isin(list(winning_nodes.values())[j], i)))
#            print('np.where(np.isin(list(winning_nodes.values())[j], i)) = ' + str(np.where(np.isin(list(winning_nodes.values())[j], i))))
            print('winning_node_2d_grid_dim = ' + str(list(winning_nodes.keys())[j]))
            winning_node_2d_grid_dim = list(winning_nodes.keys())[j]
    winning_node_number = winning_node_2d_grid_dim[1]*SOMsize[1] + winning_node_2d_grid_dim[0]
    winning_node_numbers_with_time[i] = int(winning_node_number)
counts, bins = np.histogram(winning_node_numbers_with_time, bins=25)
print('counts = ' + str(counts))
print('bins = ' + str(bins))

node_to_node_prob = np.zeros((np.prod(SOMsize), np.prod(SOMsize), len(hours)))  # 1st dim: "was" states, 2nd dim: "will be" states, 3rd dim: transition time
node_to_node_counts = np.zeros((np.prod(SOMsize), np.prod(SOMsize), len(hours)))  # 1st dim: "was" states, 2nd dim: "will be" states, 3rd dim: transition time
# For every defined length of time in hours
for h in range(0, len(hours)):
    print('hours[h] = ' + str(hours[h]))
    # For every time in the dataset except for the last number of hours given by hours(h)
    for i in range(0, len(winning_node_numbers_with_time)-hours[h]):
        # Calculate the probability that after time hours(h), a SOM node will remain the optimum node or will transition to another optimum node,
        # where each node occurs a known number of times
        print('winning_node_numbers_with_time[i] = ' + str(winning_node_numbers_with_time[i]))
        print('type(winning_node_numbers_with_time[i]) = ' + str(type(winning_node_numbers_with_time[i])))
        print('hours[h] = ' + str(hours[h]))
        print('type(hours[h]) = ' + str(type(hours[h])))
        node_to_node_prob[int(winning_node_numbers_with_time[i])][int(winning_node_numbers_with_time[i+hours[h]])][h] = node_to_node_prob[int(winning_node_numbers_with_time[i])][int(winning_node_numbers_with_time[i+hours[h]])][h] + 1/counts[int(winning_node_numbers_with_time[i])]
        # Just the counts will serve as the "forecast" histogram of transitions for each starting node "winning_nodes(i)" and time interval "hours(h)"
        node_to_node_counts[int(winning_node_numbers_with_time[i])][int(winning_node_numbers_with_time[i+hours[h]])][h] = node_to_node_counts[int(winning_node_numbers_with_time[i])][int(winning_node_numbers_with_time[i+hours[h]])][h] + 1

varname_title = 'MRMS MultiSensor 01H & 24H'

for n in range(0, np.prod(SOMsize)):
    fig = plt.figure(figsize=(27, 9))
    title_fontsize = 28
    axis_fontsize = 18
    annot_fontsize = 14
    ax = sns.heatmap(100*node_to_node_prob[n], annot=True, fmt='.1f', xticklabels=hours, yticklabels=np.arange(1, np.prod(SOMsize)+1), annot_kws={'size': annot_fontsize}, cbar_kws={'pad': 0.01})
    ax.set_xlabel('time [hours]', fontsize=axis_fontsize)
    ax.set_ylabel('future SOM node', fontsize=axis_fontsize)
    plt.yticks(rotation=0, fontsize=axis_fontsize)
    plt.xticks(rotation=45, ha='right', fontsize=axis_fontsize)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=axis_fontsize)
    ax.figure.subplots_adjust(left = 0.3)
    plt.title('Node ' + str(n+1) + ' transition probability [%] with transition period length', fontsize=title_fontsize)
    if n < 9:
        #plt.savefig('SOM_heatmap_nodetransition_node0' + str(n+1) + '_01H24Htrain_' + SOMsize_str + '_' + date_first_str + '_' + date_last_str + '_' + myregion_plot + '_test2_fixedtest2.png', bbox_inches='tight', pad_inches=0.01)
        #plt.savefig('SOM_heatmap_nodetransition_node0' + str(n+1) + '_01H24Htrain_lr25_' + SOMsize_str + '_' + date_first_str + '_' + date_last_str + '_' + myregion_plot + '_test2_fixedtest2.png', bbox_inches='tight', pad_inches=0.01)
        #plt.savefig('SOM_heatmap_nodetransition_node0' + str(n+1) + '_01H24Htrain_thresh_percentile_99th_' + SOMsize_str + '_' + date_first_str + '_' + date_last_str + '_' + myregion_plot + '_test2_fixedtest2.png', bbox_inches='tight', pad_inches=0.01)
        plt.savefig('SOM_heatmap_nodetransition_node0' + str(n+1) + '_01H24Htrain_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + SOMsize_str + '_' + date_first_str + '_' + date_last_str + '_' + myregion_plot + '_test2_fixedtest2.png', bbox_inches='tight', pad_inches=0.01)
    else:
        #plt.savefig('SOM_heatmap_nodetransition_node' + str(n+1) + '_01H24Htrain_' + SOMsize_str + '_' + date_first_str + '_' + date_last_str + '_' + myregion_plot + '_test2_fixedtest2.png', bbox_inches='tight', pad_inches=0.01)
        #plt.savefig('SOM_heatmap_nodetransition_node' + str(n+1) + '_01H24Htrain_lr25_' + SOMsize_str + '_' + date_first_str + '_' + date_last_str + '_' + myregion_plot + '_test2_fixedtest2.png', bbox_inches='tight', pad_inches=0.01)
        #plt.savefig('SOM_heatmap_nodetransition_node' + str(n+1) + '_01H24Htrain_thresh_percentile_99th_' + SOMsize_str + '_' + date_first_str + '_' + date_last_str + '_' + myregion_plot + '_test2_fixedtest2.png', bbox_inches='tight', pad_inches=0.01)
        plt.savefig('SOM_heatmap_nodetransition_node' + str(n+1) + '_01H24Htrain_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + SOMsize_str + '_' + date_first_str + '_' + date_last_str + '_' + myregion_plot + '_test2_fixedtest2.png', bbox_inches='tight', pad_inches=0.01)

