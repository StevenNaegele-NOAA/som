# Import necessary modules
import numpy as np
import xarray as xr
# from sklearn_som.som import SOM
from minisom import MiniSom
import pickle
import timeit
from sklearn.preprocessing import normalize
import datetime
import pandas
import sys


SOMsize_str = '5x5'
myregion_file = 'SC'
ari = '1yr'
precip_duration1 = '1hr'
precip_duration2 = '24hr'
varname_pickle = 'radaronly_QPE_01H_24H'  # variable name for the pickle file where output from this script will be saved
# Needs to be read in outside of the function for some reason...?
#with open('SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count.pkl', 'rb') as f:
#    varnames, myfiles, myregion, percentile_90, percentile_95, percentile_99, percentile_99_5, percentile_99_9, percentile_99_99, indices_5mm_allvars, indices_10mm_allvars, indices_25mm_allvars, indices_50mm_allvars, indices_100mm_allvars, indices_150mm_allvars, indices_200mm_allvars, indices_250mm_allvars, indices_90th_allvars, indices_95th_allvars, indices_99th_allvars, indices_99_5th_allvars, indices_99_9th_allvars, indices_99_99th_allvars, datetimes_5mm_allvars, datetimes_10mm_allvars, datetimes_25mm_allvars, datetimes_50mm_allvars, datetimes_100mm_allvars, datetimes_150mm_allvars, datetimes_200mm_allvars, datetimes_250mm_allvars, datetimes_90th_allvars, datetimes_95th_allvars, datetimes_99th_allvars, datetimes_99_5th_allvars, datetimes_99_9th_allvars, datetimes_99_99th_allvars, count_5mm_allvars, count_10mm_allvars, count_25mm_allvars, count_50mm_allvars, count_100mm_allvars, count_150mm_allvars, count_200mm_allvars, count_250mm_allvars, count_90th_allvars, count_95th_allvars, count_99th_allvars, count_99_5th_allvars, count_99_9th_allvars, count_99_99th_allvars = pickle.load(f)
#with open('SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count_99th.pkl', 'rb') as f:
#    varnames, myfiles, myregion, percentile_99_allvars, indices_99th_allvars, datetimes_99th_allvars, count_99th_allvars = pickle.load(f)
#with open('SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count_ari' + ari + '.pkl', 'rb') as f:
#    varnames, myfiles, myregion, indices_ari_allvars, datetimes_ari_allvars, count_ari_allvars = pickle.load(f)
#print('len(indices_ari' + ari + precip_duration1 + '_allvars[0]) = ' + str(len(indices_ari_allvars[0])))
#print('len(indices_ari' + ari + precip_duration2 + '_allvars[1]) = ' + str(len(indices_ari_allvars[1])))
#
#with open('SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' + myregion + '_vars_batch.pkl', 'wb') as f:  # Python 2: open(..., 'w')
##[varSOMs, dataArray, map_lats, map_lons] = mysom2_multivar(myfiles, varnames, SOMsizes, mapSize, myregion, d)
##            pickle.dump([varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles,
##                         year_month_strs, year_month_counts, mapedges, myfiles, varname, SOMsize, mapSize, myregion], f)
#        pickle.dump([varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars,
#                     year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapSize, myregion], f)

if myregion_file == 'NW':
    maptedge = 1  # top edge of the MRMS region
    mapbedge = 530#1750  # bottom "                      "
    mapledge = 1  # left "                      "
    mapredge = 900#3500  # right "                      "
elif myregion_file == 'NC':
    maptedge = 1  # note that the top left corner is 1,1
    mapbedge = 530#1750
    mapledge = 451#1751
    mapredge = 1350#5250
elif myregion_file == 'NE':
    maptedge = 1
    mapbedge = 530#1750
    mapledge = 900#3501
    mapredge = 1799#7000
elif myregion_file == 'SW':
    maptedge = 530#1751
    mapbedge = 1059#3500
    mapledge = 1
    mapredge = 900#3500
elif myregion_file == 'SC':
    maptedge = 530#1751
    mapbedge = 1059#3500
    mapledge = 451#1751
    mapredge = 1350#5250
elif myregion_file == 'SE':
    maptedge = 530#1751
    mapbedge = 1059#3500
    mapledge = 900#3501
    mapredge = 1799#7000
mapedges = [maptedge, mapbedge, mapledge, mapredge]

with open('SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' + myregion_file + '_vars_batch.pkl', 'rb') as f:  # Python 2: open(..., 'r')
    varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars, year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapSize, myregion = pickle.load(f)

precip_hist = []
for myvar in range(0, len(varnames)):
    dataArray = dataArrays[myvar]
    for mytime in range(0, len(dataArray)):
        dataArray_map = dataArray[mytime].unstack()
        precip_hist.append(dataArray_map[int((mapredge-mapledge)/2)][int((mapbedge-maptedge)/4)])  # for the SC region, get precip data from the point 1/2 in the x-direction and 1/4 of the way from the top
        
fig = plt.figure(figsize=(12, 6))
counts, bins = np.histogram(precip_hist)
plt.hist(bins[:-1], bins, weights=counts)
plt.savefig('SOM_precip_hist_test.png')
