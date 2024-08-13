# Import necessary modules
import numpy as np
import xarray as xr
#from sklearn_som.som import SOM
from minisom import MiniSom
import pickle
import timeit
from sklearn.preprocessing import normalize
import datetime
import pandas
import sys
import math


SOMsize_str = '5x5'
myregion = 'SC'
ari = '1yr'
precip_duration1 = '1hr'
precip_duration2 = '24hr'
varname_pickle = 'radaronly_QPE_01H_24H'  # variable name for the pickle file where output from this script will be saved
# Needs to be read in outside of the function for some reason...?
#with open('SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count.pkl', 'rb') as f:
#    varnames, myfiles, myregion, percentile_90, percentile_95, percentile_99, percentile_99_5, percentile_99_9, percentile_99_99, indices_5mm_allvars, indices_10mm_allvars, indices_25mm_allvars, indices_50mm_allvars, indices_100mm_allvars, indices_150mm_allvars, indices_200mm_allvars, indices_250mm_allvars, indices_90th_allvars, indices_95th_allvars, indices_99th_allvars, indices_99_5th_allvars, indices_99_9th_allvars, indices_99_99th_allvars, datetimes_5mm_allvars, datetimes_10mm_allvars, datetimes_25mm_allvars, datetimes_50mm_allvars, datetimes_100mm_allvars, datetimes_150mm_allvars, datetimes_200mm_allvars, datetimes_250mm_allvars, datetimes_90th_allvars, datetimes_95th_allvars, datetimes_99th_allvars, datetimes_99_5th_allvars, datetimes_99_9th_allvars, datetimes_99_99th_allvars, count_5mm_allvars, count_10mm_allvars, count_25mm_allvars, count_50mm_allvars, count_100mm_allvars, count_150mm_allvars, count_200mm_allvars, count_250mm_allvars, count_90th_allvars, count_95th_allvars, count_99th_allvars, count_99_5th_allvars, count_99_9th_allvars, count_99_99th_allvars = pickle.load(f)
#with open('SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count_99th.pkl', 'rb') as f:
#    varnames, myfiles, myregion, percentile_99_allvars, indices_99th_allvars, datetimes_99th_allvars, count_99th_allvars = pickle.load(f)
with open('/scratch2/BMC/wrfruc/naegele/SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count_ari' + ari + '_2021.pkl', 'rb') as f:
    varnames, myfiles, myregion, indices_ari_allvars, datetimes_ari_allvars, count_ari_allvars = pickle.load(f)
print('len(indices_ari' + ari + precip_duration1 + '_allvars[0]) = ' + str(len(indices_ari_allvars[0])))#, flush=True)
print('len(indices_ari' + ari + precip_duration2 + '_allvars[1]) = ' + str(len(indices_ari_allvars[1])))#, flush=True)

#print('len(indices_99th_allvars[0]) = ' + str(len(indices_99th_allvars[0])))  # 25768
#print('len(indices_99th_allvars[1]) = ' + str(len(indices_99th_allvars[1])))  # 24864

#def mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapSize, myregion, d):
#def mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d):
def mysom2_multivar_thresh_part2(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d, dataArrays, shared_datetimes, datetimes_common):
    print('Starting mysom function (part 2) ...')#, flush=True)

    maptedge, mapbedge, mapledge, mapredge = mapedges

        # Stack (horizontally) the vector maps of each variable at each time, so that each time has one long vector map of all variables
        # >>> blah1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # >>> blah2 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
        # >>> big_blah = np.hstack((blah1, blah2))
        # >>> print(big_blah)
        # [[ 1  2  3 10 11 12]
        #  [ 4  5  6 13 14 15]
        #  [ 7  8  9 16 17 18]]
        # >>> all_blahs = []
        # >>> all_blahs.append(blah1)
        # >>> all_blahs.append(blah2)
        # >>> print(all_blahs)
        # [array([[1, 2, 3],
        #        [4, 5, 6],
        #        [7, 8, 9]]), array([[10, 11, 12],
        #        [13, 14, 15],
        #        [16, 17, 18]])]
        # >>> big_blah_v2 = np.hstack(all_blahs)
        # >>> print(big_blah_v2)
        # [[ 1  2  3 10 11 12]
        #  [ 4  5  6 13 14 15]
        #  [ 7  8  9 16 17 18]]
        #print(dataArrays)
    dataArrays_orig = dataArrays
#        dataArrays_allSOMs.append(dataArrays)
        #print(vectorOfMap)
        #print(type(vectorOfMap))  # xarray dataarray
        #print(dataArray)
        #print(type(dataArray))  # list (still?)
        #print(dataArrays[0])
        #print(type(dataArrays[0]))  # numpy ndarray
        #print(dataArrays)
        #print(type(dataArrays))  # list
        #dataArrays = np.array(dataArrays)
    da_counter = -1
    dataArrays_normalized = []
    for dataArray in dataArrays:
        da_counter = da_counter + 1
#        for x in range(0, len(dataArrays[da_counter])):
#            dataArrays[da_counter][x] = np.ma.array(dataArrays[da_counter][x])  # this dataset contains fill values, so in order to normalize and train on the data, the fill values need to be masked
#            for y in range(0, len(dataArrays[da_counter][x])):
#                if np.isnan(dataArrays[da_counter][x][y]) or dataArrays[da_counter][x][y] == -999 or dataArrays[da_counter][x][y] == -9999:
#                    dataArrays[da_counter][x][y] = np.ma.masked
        print('len(dataArray) = ' + str(len(dataArray)))  # 12533 --> 25768 --> 9934
        print('len(dataArray[0]) = ' + str(len(dataArray[0])))  #                  4770
#        np.set_printoptions(threshold=sys.maxsize)
        print('dataArray[0] = ' + str(dataArray[0]))  # 
#        np.set_printoptions(threshold=False)
#        print('dataArray[-1] = ' + str(dataArray[-1]))  # [0. 0. 0. ... 0. nan nan]
#        print('len(shared_indices) = ' + str(len(shared_indices)))  # 25938
#        print('shared_indices[-1] = ' + str(shared_indices[-1]))  # 26181
#        dataArray = dataArray[shared_indices]  # IndexError: index 32772 is out of bounds for axis 0 with size 9934  --> 32772 is first of shared_indices
        print('len(shared_datetimes) = ' + str(len(shared_datetimes)))  # 25938 --> 9906
        print('shared_datetimes[-1] = ' + str(shared_datetimes[-1]))  # 26181
#        both = set(datetimes_ari_allvars[myvar]).intersection(shared_datetimes)  # get the 
#        shared_indices = [datetimes_ari_allvars[da_counter].index(x) for x in set(shared_datetimes)]  # get the indices (for the current variable's dataArray) of the datetimes shared among all variables  #ValueError: '20191003-18' is not in list
#        shared_indices = [datetimes_allvars[da_counter].index(x) for x in set(shared_datetimes)]  # get the indices (for the current variable's dataArray) of the datetimes shared among all variables  #ValueError: '20191003-18' is not in list
        shared_indices = [datetimes_common.index(x) for x in set(shared_datetimes)]  # get the indices (for the current variable's dataArray) of the datetimes shared among all variables  #ValueError: '20191003-18' is not in list
        dataArray = dataArray[shared_indices]  # IndexError: index 32772 is out of bounds for axis 0 with size 9934  --> 32772 is first of shared_indices
        print('NaN count for dataArray[' + str(da_counter) + '] = ' + str(np.count_nonzero(np.isnan(dataArray))))
#        dataArray_normalized = []
#        dataArray_normalized = dataArray

#vectorOfMap = np.ma.array(vectorOfMap)  # this dataset contains fill values, so in order to normalize and train on the data, the fill values need to be masked
#for x in range(0, len(vectorOfMap)):
#    if np.isnan(vectorOfMap[x]) or vectorOfMap[x] == -999 or vectorOfMap[x] == -9999:
#        vectorOfMap[x] = np.ma.masked

##        for gridpoint in range(0, np.prod(int((mapbedge-maptedge)/10),int((mapredge-mapledge)/10))):
#        for gridpoint in range(0, len(dataArray[0])):
##            gridpoint_normalized = normalize(dataArray[:][gridpoint][~np.isnan(dataArray[:][gridpoint])])
##            gridpoint_normalized = (np.array(dataArray)[:,gridpoint] - np.nanmin(np.array(dataArray)[:,gridpoint])) / (np.nanmax(np.array(dataArray)[:,gridpoint]) - np.nanmin(np.array(dataArray)[:,gridpoint]))
#            alltimes_gridpoint_min = np.nanmin(np.array(dataArray)[:,gridpoint])
#            alltimes_gridpoint_max = np.nanmax(np.array(dataArray)[:,gridpoint])
#            for mytime in range(0, len(dataArray)):
#                onetime_gridpoint = np.array(dataArray)[mytime,gridpoint] 
##                dataArray_normalized[mytime,gridpoint] = (np.array(dataArray)[mytime,gridpoint] - np.nanmin(np.array(dataArray)[:,gridpoint])) / (np.nanmax(np.array(dataArray)[:,gridpoint]) - np.nanmin(np.array(dataArray)[:,gridpoint]))
#                if not np.isnan(onetime_gridpoint):
#                    dataArray_normalized[mytime,gridpoint] = 1#(onetime_gridpoint - alltimes_gridpoint_min) / (alltimes_gridpoint_max - alltimes_gridpoint_min)
#                elif np.isnan(onetime_gridpoint):
#                    dataArray_normalized[mytime,gridpoint] = onetime_gridpoint
##        dataArray = normalize(dataArray[~np.isnan(dataArray)], norm='l1', axis=1)  # L1 normalization each row independently, and normalize each sample (across time?) instead of each feature (across space?)
##            dataArray_normalized.append(gridpoint_normalized)

        alltimes_gridpoint_min = np.nanmin(dataArray, axis=0)
        alltimes_gridpoint_max = np.nanmax(dataArray, axis=0)
        dataArray_normalized = [(onetime_onepoint - alltimes_gridpoint_min) / (alltimes_gridpoint_max - alltimes_gridpoint_min) for onetime_onepoint in dataArray]
#        dataArray_normalized = np.ma.array(dataArray_normalized)
#        dataArray_normalized = [[x if not np.isnan(x) else x == np.ma.masked for x in y] for y in dataArray_normalized]

        dataArrays_normalized.append(xr.DataArray(dataArray_normalized))
    print('len(dataArrays_normalized) = ' + str(len(dataArrays_normalized)))  # 2
    print('len(dataArrays_normalized[0]) = ' + str(len(dataArrays_normalized[0])))  # 4770
    print('len(dataArrays_normalized[0][0]) = ' + str(len(dataArrays_normalized[0][0])))  # 9906
#    print('np.nanmin(dataArrays_normalized) = ' + str(np.nanmin(dataArrays_normalized)))
#    print('np.nanmax(dataArrays_normalized) = ' + str(np.nanmax(dataArrays_normalized)))

    print(dataArrays)
    print('NaN count for dataArrays = ' + str(np.count_nonzero(np.isnan(dataArrays))))
        #print(dataArrays[0])
        #print(type(dataArrays[0]))  # numpy ndarray
        #print(dataArrays)
        #print(type(dataArrays))  # list
#    dataArray_train = np.hstack(dataArrays)  # dataArrays needs to be an array for some reason? (even though in testing a list was fine)
    dataArray_train = np.hstack(dataArrays_normalized)  # dataArrays needs to be an array for some reason? (even though in testing a list was fine)
    print(len(dataArray_train))  # 9906
    print(len(dataArray_train[0]))  # 9540 (2*90*53)
#    print(dataArray_train.shape)
    print('NaN count for dataArray_train = ' + str(np.count_nonzero(np.isnan(dataArray_train))))
        #dataArray_train = []
        #for mytime in range(0, len(dataArrays[0])):
        #    dataArray_train.append(
#        dataArray_train_allSOMs.append(dataArray_train)

    print('Configuring and training SOM ...')
#        varSOM = MiniSom(SOMsize[0], SOMsize[1], np.prod(mapSize), sigma=0.3, learning_rate=0.5)
#    varSOM = MiniSom(SOMsize[0], SOMsize[1], len(varnames)*np.prod(mapSize), sigma=1.75, learning_rate=0.125)  # optimal? [3x4, sigma=1.75, learning_rate=0.125, epochs=125]
#    varSOM = MiniSom(SOMsize[0], SOMsize[1], len(varnames)*np.prod([math.ceil((mapredge-mapledge)/10), math.ceil((mapbedge-maptedge)/10)]), sigma=1.75, learning_rate=0.125)  # optimal? [3x4, sigma=1.75, learning_rate=0.125, epochs=125]
    varSOM = MiniSom(SOMsize[0], SOMsize[1], len(varnames)*np.prod([math.ceil((mapredge-mapledge)/10), math.ceil((maptedge-mapbedge)/10)]), sigma=1.75, learning_rate=0.125)  # optimal? [3x4, sigma=1.75, learning_rate=0.125, epochs=125]
#    varSOM = SOM(SOMsize[0], SOMsize[1], dim=len(varnames)*np.prod([math.ceil((mapredge-mapledge)/10), math.ceil((mapbedge-maptedge)/10)]), sigma=1.75, lr=0.125, max_iter=500)
#    varSOM = SOM(SOMsize[0], SOMsize[1], dim=len(varnames)*np.prod([math.ceil((mapredge-mapledge)/10), math.ceil((mapbedge-maptedge)/10)]), sigma=1.99, lr=0.975, max_iter=500)
#    varSOM = MiniSom(SOMsize[0], SOMsize[1], len(varnames)*np.prod(mapSize), sigma=1.75, learning_rate=0.25)  # 99th percentile & incr learning rate to resolve nodes only representing first & last months
#    varSOM = MiniSom(SOMsize[0], SOMsize[1], len(varnames)*np.prod(mapSize), sigma=1.75, learning_rate=0.25)  # increasing learning rate from "optimal" value in an attempt to figure out why 
                                                                                                              # some nodes in larger grids have a few cases for the first month and then none afterward
#    dataArray_train = np.ma.array(dataArray_train)
#    dataArray_train = xr.DataArray(dataArray_train)#.to_masked_array(copy=False)
#    dataArray_train = [[x if not np.isnan(x) else x == np.ma.masked for x in y] for y in dataArray_train]
#    dataArray_train = [[x if not np.isnan(x) else x == 0 for x in y] for y in dataArray_train]
#        varSOM = MiniSom(SOMsize[0], SOMsize[1], np.prod(mapSize), sigma=1.0, learning_rate=0.125)  # alt sigma [spread of neighborhood function] (which is actually the default)
#        varSOM.train_batch(dataArray_train, 1000)
    varSOM.train_batch(~np.isnan(dataArray_train), 500)  # ValueError: Received 9540 features, expected 9256. 2*90*53=9540 2*89*52=9256  ValueError: Received 19812 features, expected 9540.
#    print(xr.DataArray.to_masked_array(dataArray_train))
#    print(xr.DataArray(dataArray_train).to_masked_array(copy=False)[0])
    #print(dataArray_train[0])
    # [<xarray.DataArray ()>
    # array(0., dtype=float32), <xarray.DataArray ()>
    # array(0., dtype=float32), <xarray.DataArray ()>
    # ...
    # array(0.04137931, dtype=float32), <xarray.DataArray ()>
    # array(False), <xarray.DataArray ()>
    # array(False)]
    # --> 9540 lines
#    varSOM.fit(xr.DataArray.to_masked_array(dataArray_train))
#    varSOM.fit(xr.DataArray(dataArray_train).to_masked_array(copy=False))
#    varSOM.fit(dataArray_train)
#    varSOM.fit(xr.DataArray(dataArray_train))
#    predictions = varSOM.predict(xr.DataArray(dataArray_train))
#    predictions = varSOM.predict(xr.DataArray(dataArray_train).to_masked_array(copy=False))
#    predictions = varSOM.predict(dataArray_train)
#    predictions = varSOM.predict(xr.DataArray(dataArray_train))
#        varSOMs.append(varSOM)
#    print('predictions: ')
#    print(predictions)

    # Save important variables to a file
    print('Saving important variables to a file and returning to main script ...')
#    with open('SOM_' + SOMsize_str + '_thresh_percentile_99th_lr25_' + varname_pickle + '_' + myregion + '_vars_batch.pkl', 'wb') as f:  # Python 2: open(..., 'w')
    with open('/scratch2/BMC/wrfruc/naegele/SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' + myregion + '_vars_batch_part2_zeros_2021.pkl', 'wb') as f:  # Python 2: open(..., 'w')
#[varSOMs, dataArray, map_lats, map_lons] = mysom2_multivar(myfiles, varnames, SOMsizes, mapSize, myregion, d)
#            pickle.dump([varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles,
#                         year_month_strs, year_month_counts, mapedges, myfiles, varname, SOMsize, mapSize, myregion], f)
#        pickle.dump([varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars,
#                     year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapSize, myregion], f)
#        pickle.dump([varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars,
#                     year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapedges, myregion], f)
        pickle.dump([varSOM, dataArrays_orig, dataArray_train, mapedges, myfiles, varnames, SOMsize, mapedges, myregion], f)
#        pickle.dump([varSOM, predictions, dataArrays_orig, dataArray_train, mapedges, myfiles, varnames, SOMsize, mapedges, myregion], f)
        # To open the file later
        # with open('objs.pkl', 'rb') as f:  # Python 2: open(..., 'r') Python 2: open(..., 'rb')
        #     obj0, obj1, obj2, ... = pickle.load(f)

#    return varSOM, dataArray, varSOM_nonzero, dataArray_nonzero, map_lats, map_lons, months_doubles, years_doubles, \
#    return varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles, \
#        year_month_strs, year_month_counts, mapedges
#    return varSOMs, dataArray_train_allSOMs, dataArrays_allSOMs, map_lats, map_lons
#    return varSOM, dataArray_train, dataArrays, percentile_99_allvars, indices_99th_allvars, count_99th_allvars, map_lats, map_lons
    return varSOM, dataArray_train
#    return varSOM, predictions, dataArray_train


