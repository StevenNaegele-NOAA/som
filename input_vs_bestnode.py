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
import math
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy


SOMsize_str = '3x2'
myregion_file = 'SE'
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
    maptedge = 1059#1  # top edge of the MRMS region
    mapbedge = 530#1750  # bottom "                      "
    mapledge = 1  # left "                      "
    mapredge = 900#3500  # right "                      "
elif myregion_file == 'NC':
    maptedge = 850#1059#1  # note that the top left corner is 1,1
    mapbedge = 500#530#1750
    mapledge = 551#451#1751
    mapredge = 1300#1350#5250
elif myregion_file == 'NE':
    maptedge = 850#1059#1
    mapbedge = 701#530#1750
    mapledge = 1426#900#3501
    mapredge = 1650#1799#7000
elif myregion_file == 'SW':
    maptedge = 650#530#1751
    mapbedge = 351#1#1059#3500
    mapledge = 501#1
    mapredge = 750#900#3500
elif myregion_file == 'SC':
    maptedge = 530#550#500#530#1751
    mapbedge = 1#1059#3500
    mapledge = 551#451#1751
    mapredge = 1350#1400#1050#1500#1350#5250
elif myregion_file == 'SE':
    maptedge = 530#1751
    mapbedge = 1#1059#3500
    mapledge = 900#3501
    mapredge = 1799#7000
elif myregion_file == 'MA':  # Mid-Atlantic
    maptedge = 700
    mapbedge = 451
    mapledge = 1301
    mapredge = 1550
mapedges = [maptedge, mapbedge, mapledge, mapredge]

#with open('SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' + myregion_file + '_vars_batch.pkl', 'rb') as f:  # Python 2: open(..., 'r')
#    varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars, year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapSize, myregion = pickle.load(f)
#with open('/scratch2/BMC/wrfruc/naegele/SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' + myregion_file + '_vars_batch_newdomains_sklearn.pkl', 'rb') as f:  # Python 2: open(..., 'r')
#    varSOM, predictions, dataArrays_orig, dataArray_train, mapedges, myfiles, varnames, SOMsize, mapedges, myregion = pickle.load(f)
### /scratch2/BMC/wrfruc/naegele/SOM_3x2_thresh_ari1yr1hr_ari1yr24hr_radaronly_QPE_01H_24H_SE_vars_batch_newdomains_sklearn_2021.pkl
with open('/scratch2/BMC/wrfruc/naegele/SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' + myregion_file + '_vars_batch_newdomains_sklearn_2021.pkl', 'rb') as f:  # Python 2: open(..., 'r')
    varSOM, weights, predictions, dataArrays_orig, dataArray_train, mapedges, myfiles, varnames, SOMsize, mapedges, myregion = pickle.load(f)

#precip_hist = []
#for myvar in range(0, len(varnames)):
#    dataArray = dataArrays[myvar]
#    for mytime in range(0, len(dataArray)):
#        dataArray_map = dataArray[mytime].unstack()
#        precip_hist.append(dataArray_map[int((mapredge-mapledge)/2)][int((mapbedge-maptedge)/4)])  # for the SC region, get precip data from the point 1/2 in the x-direction and 1/4 of the way from the top
        
for myvar in range(0, len(varnames)):
    dataArray = dataArrays_orig[myvar]
    node_count = -1
    my_var_cmap = 'Blues'
    my_var_vmin = 0
    if myvar == 0:
        my_var_vmax = 5#1#0.5#1#5#25#5
        my_var_levels = [0,1,2,3,4,5]#[0,0.2,0.4,0.6,0.8,1]#[0,0.1,0.2,0.3,0.4,0.5]#[0,0.2,0.4,0.6,0.8,1]#[0,1,2,3,4,5]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
        my_var_tick_levels = [0,1,2,3,4,5]#[0,0.2,0.4,0.6,0.8,1]#[0,0.1,0.2,0.3,0.4,0.5]#[0,0.2,0.4,0.6,0.8,1]#[0,1,2,3,4,5]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
    elif myvar == 1:
        my_var_vmax = 25#5#10#20#10#25#5
        my_var_levels = [0,5,10,15,20,25]#[0,1,2,3,4,5]#[0,2,4,6,8,10]#[0,4,8,12,16,20]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
        my_var_tick_levels = [0,5,10,15,20,25]#[0,1,2,3,4,5]#[0,2,4,6,8,10]#[0,4,8,12,16,20]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
    my_var_cbarextend = 'max'
    my_var_cbarlabel = 'liquid precipitation [mm]'
    fig = plt.figure(figsize=(12,6))
#    for mytime in range(0, len(dataArray)):
    for iSOM in range(0, SOMsize[1]):
        for jSOM in range(0, SOMsize[0]):
            node_count = node_count + 1
            print('>>> Creating best-matched precip plot for node ' + str(node_count + 1))
            winning_node_indices = np.where(predictions == node_count)
            domain_avg_diff = 999
#            print('predictions = ' + str(predictions))
            print('>>> len(winning_node_indices) = ' + str(len(winning_node_indices[0])))
#            print('winning_node_indices = ' + str(winning_node_indices))
            node_avg = np.nanmean(dataArray[winning_node_indices[0]], axis=0)
            print('>>> np.nanmean(dataArray[winning_node_indices], axis=0) = ' + str(node_avg))
            for mytime in range(0, len(winning_node_indices[0])):
                if abs(np.nanmean(dataArray[mytime] - node_avg)) < domain_avg_diff:
                    domain_avg_diff = abs(np.nanmean(dataArray[mytime] - node_avg))
                    best_time = mytime
                    dataArray_node_map = dataArray[mytime].reshape(-1, math.ceil((mapredge-mapledge)/10), math.ceil((maptedge-mapbedge)/10))
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
            cart_proj = ccrs.PlateCarree()
            ax = plt.subplot(SOMsize[1], SOMsize[0], node_count + 1, projection=cart_proj)
            example_full_filename = '/scratch1/BMC/wrfruc/ejames/climo/mrms/01h/202212/mrms_radaronly_2022123123.nc'
            testvarread = xr.open_dataset(example_full_filename, engine='netcdf4')
            map_lats = testvarread['latitude'][np.arange(mapbedge, maptedge, 10), np.arange(mapledge, mapredge, 10)]  # lat_0 is 3500 long
            map_lons = testvarread['longitude'][np.arange(mapbedge, maptedge, 10), np.arange(mapledge, mapredge, 10)]
            print('>>> domain_avg_diff = ' + str(domain_avg_diff))
            print('>>> best_time = ' + str(best_time))
            print('>>> dataArray_node_map.shape = ' + str(dataArray_node_map.shape))
            ax_ = ax.contourf(map_lons, map_lats, np.transpose(np.squeeze(dataArray_node_map)), cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
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
    plt.suptitle(varnames[myvar] + ' -- ' + 'best-matched time (index = ' + str(best_time) + ', domain-avg diff = ' + str(domain_avg_diff) + ')', x=0.5125, y=0.9125, fontsize=14)
    plt.savefig('SOM_' + SOMsize_str + '_' + varnames[myvar] + '_bestmatchedtime.png')

    node_count = -1
    fig = plt.figure(figsize=(12,6))
    for iSOM in range(0, SOMsize[1]):
        for jSOM in range(0, SOMsize[0]):
            node_count = node_count + 1
            print('weights[node_count].shape @ node ' + str(node_count + 1) + ' = ' + str(weights[node_count].shape))
            ax = plt.subplot(SOMsize[1], SOMsize[0], node_count + 1, projection=cart_proj)
            weights_myvar_num = int(len(weights[node_count])/len(varnames))
            weights_myvar_begin = int(weights_myvar_num*myvar)
            weights_myvar_end = int(weights_myvar_num*(myvar+1))
            print('weights_myvar_num, weights_myvar_begin, weights_myvar_end = ' + str(weights_myvar_num) + ', ' + str(weights_myvar_begin) + ', ' + str(weights_myvar_end))
            weights_myvar = weights[node_count][weights_myvar_begin:weights_myvar_end]
            print('weights_myvar.shape = ' + str(weights_myvar.shape))
            print('weights_myvar = ' + str(weights_myvar))
            print('np.squeeze(np.transpose(weights_myvar.reshape(-1, math.ceil((mapredge-mapledge)/10), math.ceil((maptedge-mapbedge)/10)))).shape = ' + str(np.squeeze(np.transpose(weights_myvar.reshape(-1, math.ceil((mapredge-mapledge)/10), math.ceil((maptedge-mapbedge)/10)))).shape))
            print('np.squeeze(np.transpose(weights_myvar.reshape(-1, math.ceil((mapredge-mapledge)/10), math.ceil((maptedge-mapbedge)/10)))) = ' + str(np.squeeze(np.transpose(weights_myvar.reshape(-1, math.ceil((mapredge-mapledge)/10), math.ceil((maptedge-mapbedge)/10))))))
            ax_ = ax.contourf(map_lons, map_lats, np.squeeze(np.transpose(weights_myvar.reshape(-1, math.ceil((mapredge-mapledge)/10), math.ceil((maptedge-mapbedge)/10)))), cmap='Greens', vmin=-0.1, vmax=0.2,
                                              levels=[-0.1,-0.05,0,0.05,0.1,0.15,0.2], transform=ccrs.PlateCarree(), extend='both')
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
    cbar = fig.colorbar(ax_, cax=cax, orientation='horizontal', extend='both',
                        ticks=[-0.1,-0.05,0,0.05,0.1,0.15,0.2], label='SOM node weights []')#, pad=0.05)
    plt.suptitle(varnames[myvar] + ' -- ' + 'SOM node weights', x=0.5125, y=0.9125, fontsize=14)
    plt.savefig('SOM_' + SOMsize_str + '_' + varnames[myvar] + '_nodeweights.png')


#fig = plt.figure(figsize=(12, 6))
#counts, bins = np.histogram(precip_hist)
#plt.hist(bins[:-1], bins, weights=counts)
#plt.savefig('SOM_precip_hist_test.png')
