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
from datetime import datetime, timedelta
import seaborn as sns
from minisom import MiniSom
import pandas

myregion_plot = str(sys.argv[1])
print('My region =', myregion_plot)

if myregion_plot == 'NW':
    maptedge = 1059#1  # top edge of the MRMS region
    mapbedge = 530#1750  # bottom "                      "
    mapledge = 1  # left "                      "
    mapredge = 900#3500  # right "                      "
elif myregion_plot == 'NC':
    maptedge = 850#1059#1  # note that the top left corner is 1,1
    mapbedge = 500#530#1750
    mapledge = 551#451#1751
    mapredge = 1300#1350#5250
elif myregion_plot == 'NE':
    maptedge = 850#1059#1
    mapbedge = 701#530#1750
    mapledge = 1426#900#3501
    mapredge = 1650#1799#7000
elif myregion_plot == 'SW':
    maptedge = 650#530#1751
    mapbedge = 351#1#1059#3500
    mapledge = 501#1
    mapredge = 750#900#3500
elif myregion_plot == 'SC':
    maptedge = 530#550#500#530#1751
    mapbedge = 1#1059#3500
    mapledge = 551#451#1751
    mapredge = 1350#1400#1050#1500#1350#5250
elif myregion_plot == 'SE':
    maptedge = 530#1751
    mapbedge = 1#1059#3500
    mapledge = 900#3501
    mapredge = 1799#7000
elif myregion_plot == 'MA':  # Mid-Atlantic
    maptedge = 700
    mapbedge = 451
    mapledge = 1301
    mapredge = 1550

mapSize = [53, 90]  # size of each of the future SOM maps

ari = '1yr'
precip_duration1 = '1hr'
precip_duration2 = '24hr'
varname_pickle = 'radaronly_QPE_01H_24H'  # variable name for the pickle file where output from this script will be saved

mySOMsizes = [[3,2]]#[[5,5]]#[[2,2], [3,2], [3,3], [4,3], [4,4], [5,4], [5,5]]

SOMcount = -1

for mySOMsize in mySOMsizes:

    SOMcount = SOMcount + 1

    SOMsize_str = str(mySOMsize[0]) + 'x' + str(mySOMsize[1])

    print('My SOMsize = ', SOMsize_str)

#with open('SOM_3x2_MultiSensor_QPE_01H_Pass2_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
#with open('SOM_4x3_MultiSensor_QPE_01H_Pass2_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
#    with open('SOM_' + SOMsize_str + '_learningrate25_MultiSensor_QPE_01H_24H_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
##    with open('SOM_' + SOMsize_str + '_thresh_percentile_99th_MultiSensor_QPE_01H_24H_' + myregion_plot + '_vars_batch.pkl', 'rb') as f: 
##    varSOM, dataArray, varSOM_nonzero, dataArray_nonzero, map_lats, map_lons, months_doubles, years_doubles, \
##    varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles, \
#        varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars, \
#        year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapSize, myregion = pickle.load(f)
##    year_month_strs, year_month_counts, mapedges, myfiles, varname, SOMsize, mapSize, myregion = pickle.load(f)
#    with open('/scratch2/BMC/wrfruc/naegele/SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' + myregion_plot + '_vars_batch_newdomains_sklearn_2021_lr975_6hr.pkl', 'rb') as f:  # Python 2: open(..., 'r')
    with open('/scratch2/BMC/wrfruc/naegele/SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' + myregion_plot + '_vars_batch_newdomains_sklearn.pkl', 'rb') as f:  # Python 2: open(..., 'r')
#        varSOM, weights, predictions, dataArrays_orig, dataArray_train, mapedges, myfiles, varnames, SOMsize, mapedges, myregion = pickle.load(f)
        varSOM, predictions, dataArrays_orig, dataArray_train, mapedges, myfiles, varnames, SOMsize, mapedges, myregion = pickle.load(f)

    with open('/scratch1/BMC/wrfruc/naegele/SOM_3x2_multivar_' + myregion + '_precip_thresh_count_ari' + ari + '_6hr.pkl', 'rb') as f:#'_2021.pkl', 'rb') as f:
        varnames, myfiles, myregion, indices_ari_allvars, datetimes_ari_allvars, count_ari_allvars = pickle.load(f)
    print('datetimes_ari_allvars[0][0] = ' + str(datetimes_ari_allvars[0][0]))  # 20150220-18

    sample_filename = '/scratch1/BMC/wrfruc/ejames/climo/mrms/202212/mrms_radaronly_2022123118.nc'
    testvarread = xr.open_dataset(sample_filename, engine='netcdf4')#, chunks=2000)  # load the dataset into dask arrays, of size 2000 in each dimension
    map_lats = testvarread['latitude'][np.arange(mapbedge, maptedge, 10), np.arange(mapledge, mapredge, 10)]  # lat_0 is 3500 long
    map_lons = testvarread['longitude'][np.arange(mapbedge, maptedge, 10), np.arange(mapledge, mapredge, 10)]

#    winning_nodes = varSOM.win_map(dataArray_train, return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
#                                                                          # note that the SOM node indices are [col, row], and I will read through cols first

    varcount = -1
    varnames = ['01hQPE', '24hQPE']

    datetimes_allvars = []
###
    sample_filename = '/scratch1/BMC/wrfruc/ejames/climo/mrms/202212/mrms_radaronly_2022123118.nc'
    filename_first = myfiles[0][0]#[varcount][0]
    print('filename_first = ' + str(filename_first))
#        date_first = filename_first.split('/')[-1].split('.')[1][3:10+1]
    date_first = filename_first.split('/')[-1].split('_')[-1][0:9+1]  # e.g. '2022123100'
    date_first = date_first[0:7+1] + '-' + '00'#date_first[8:9+1]
    print('date_first = ' + str(date_first))
    filename_last = myfiles[0][-1]#[varcount][-1]
    print('filename_last = ' + str(filename_last))
#        date_last = filename_last.split('/')[-1].split('.')[1][3:10+1]
    date_last = filename_last.split('/')[-1].split('_')[-1][0:9+1]
    date_last = date_last[0:7+1] + '-' + '18'#date_last[8:9+1]
    print('date_last = ' + str(date_last))
    datetimes_full = pandas.date_range(date_first,date_last,freq='6h')
#    print('datetimes_full [pre-format] = ' + str(datetimes_full))
    datetimes_full_yearmon = datetimes_full.format(formatter=lambda x: x.strftime('%Y%m'))
#    datetimes_full_indices = np.range(0, len(datetimes_full))
#    print('len(datetimes_full_indices) = ' + str(len(datetimes_full_indices)))
    datetimes_full = datetimes_full.format(formatter=lambda x: x.strftime('%Y%m%d-%H'))
#    print('datetimes_full [post-format] = ' + str(datetimes_full))
###
###
#    datetime_6hr_indices_allvars = []
#    datetimes_allvars_6hr = []
#    for myvarname in varnames:
#        varcount = varcount + 1
#        myfile_count = -1
#        datetimes_myvar = []
#        datetimes_myvar_6hr = []
#        datetime_6hr_indices_myvar = []
#        for myfile in myfiles[varcount]:
###            myyear = myfile.split('/')[-1].split('.')[1][3:6+1]
##            myyear = myfile.split('/')[-1].split('_')[-1][0:3+1]
###            mymonthname = d[myfile.split('/')[-1].split('.')[1][7:8+1]]
##            mymonthname = d[myfile.split('/')[-1].split('_')[-1][4:5+1]]
##            year_month_str_ = myyear + '-' + mymonthname
##            year_month_strs_all.append(year_month_str_)
#            myfile_count = myfile_count + 1
#            mydatetime = myfile.split('/')[-1].split('_')[-1][0:9+1]
#            datetimes_myvar.append(mydatetime[0:7+1] + '-' + mydatetime[8:9+1])
#            if mydatetime[0:7+1]+'-'+mydatetime[8:9+1] in datetimes_full:
##                print('myfile_count = ' + str(myfile_count))
##                print('mydatetime = ' + str(mydatetime))
#                datetime_6hr_indices_myvar.append(myfile_count)
#                datetimes_myvar_6hr.append(mydatetime[0:7+1] + '-' + mydatetime[8:9+1])
#        datetimes_allvars.append(datetimes_myvar)
#        datetime_6hr_indices_allvars.append(datetime_6hr_indices_myvar)
#        datetimes_allvars_6hr.append(datetimes_myvar_6hr)
#        print('datetimes_myvar for myvarname=' + myvarname + ' = ' + str(datetimes_myvar))
#        print('datetime_6hr_indices_myvar for myvarname=' + myvarname + ' = ' + str(datetime_6hr_indices_myvar))
#        print('datetimes_myvar_6hr for myvarname=' + myvarname + ' = ' + str(datetimes_myvar_6hr))

    shared_datetimes = set(datetimes_ari_allvars[0])  # start the list of datetimes where the threshold is met for at least one variable, with the datetimes for the first variable (converted to a set)
    print('len(datetimes_ari_allvars[0]) = ' + str(len(datetimes_ari_allvars[0])))
    print('datetimes_ari_allvars[0] = ' + str(datetimes_ari_allvars[0]))
    for myvar in range(1, len(varnames)):  # loop through the remaining variables
#        indices_in_2nd_set_but_not_1st = set(indices_99th_allvars[myvar]) - shared_indices  # subtract the set of indices for myvar from the set of indices for the variables represented by shared_indices
#        indices_in_2nd_set_but_not_1st = set(indices_ari_allvars[myvar]) - shared_indices  # subtract the set of indices for myvar from the set of indices for the variables represented by shared_indices
        print('len(datetimes_ari_allvars[' + str(myvar) + ']) = ' + str(len(datetimes_ari_allvars[myvar])))
        print('datetimes_ari_allvars[' + str(myvar) + '] = ' + str(datetimes_ari_allvars[myvar]))
        datetimes_in_2nd_set_but_not_1st = set(datetimes_ari_allvars[myvar]) - shared_datetimes  # subtract the set of datetimes for myvar from the set of datetimes for the variables represented by shared_datetimes
                                                                                            # when myvar = 1 (var2) shared_datetimes represents datetimes from var1 that meet the threshold
                                                                                            # when myvar = 2 (var3) shared_datetimes represents datetimes from var1 and var2 that meet the threshold, without duplicates
                                                                                            # when myvar = 3 (var4) shared_datetimes represents datetimes from var1, var2, and var3 that meet the threshold, without duplicates, etc.
#        shared_indices = set(sorted(list(shared_indices) + list(indices_in_2nd_set_but_not_1st)))  # add the indices from myvar not already in shared_indices to shared_indices
        shared_datetimes = set(sorted(list(shared_datetimes) + list(datetimes_in_2nd_set_but_not_1st)))  # add the datetimes from myvar not already in shared_datetimes to shared_datetimes
                                                                                              # when myvar = 1 (var2) shared_datetimes is overwritten from var1 datetimes that meet the threshold to var1 and var2 datetimes that meet the threshold
                                                                                              # when myvar = 1 (var2) shared_datetimes is overwritten from var1 and var2 datetimes that meet the threshold to var1, var2, and var3 datetimes that meet the threshold, etc.
#    shared_indices = sorted(list(shared_indices))
#    print('len(shared_indices) = ' + str(len(shared_indices)))  # 25768; 24864
#    print('shared_indices = ' + str(shared_indices))
    shared_datetimes = sorted(list(shared_datetimes))
    print('len(shared_datetimes) = ' + str(len(shared_datetimes)))  # 25768; 24864  --> 11170, which is just 6-hourly datetimes 
    print('shared_datetimes = ' + str(shared_datetimes))

###
#    np.set_printoptions(threshold=sys.maxsize)
#    print('datetimes_ari_allvars[0] = ' + str(datetimes_ari_allvars[0]))
#    print('len(datetimes_ari_allvars[0]) = ' + str(len(datetimes_ari_allvars[0])))  # 8542
#    print('datetimes_ari_allvars[1] = ' + str(datetimes_ari_allvars[1]))
#    print('len(datetimes_ari_allvars[1]) = ' + str(len(datetimes_ari_allvars[1])))  # 9614
#    np.set_printoptions(threshold=False)

    varcount = -1
    for myvarname in varnames:

        print('My varname = ', myvarname)

        varcount = varcount + 1

### fix so that 1hr precip is only used in 6hr intervals
##        print('myfiles[varcount] = ' + str(myfiles[varcount]))
#        datetimes_myvar = []
#        for fileCount in range(0,len(myfiles[myvar])):
#            myfile_myvar = myfiles[myvar][fileCount]  # files for any variable, including the first one (in which case no files will be missing)
##            date_myvar = myfile_myvar.split('/')[-1].split('.')[1][3:13+1]
#            date_myvar = myfile_myvar.split('/')[-1].split('_')[-1][0:9+1]
##            date_myvar = date_myvar[0:3+1] + date_myvar[4:5+1] + date_myvar[6:7+1] + '-' + date_myvar[9:10+1]
#            date_myvar = date_myvar[0:3+1] + date_myvar[4:5+1] + date_myvar[6:7+1] + '-' + date_myvar[8:9+1]
##            print('date_myvar = ' + date_myvar + ', current time = ' + str(datetime.datetime.now()))
#            datetimes_myvar.append(date_myvar)
#        datetimes_allvars.append(datetimes_myvar)
#    for myvar in range(0, len(varnames)):  # STILL NEEDS TO BE MORE GENERALIZED
#        dataArray = []
#        datetimes_common = []
#        for dateCount in range(0, len(datetimes_full)):
##                print(datetimes_full[dateCount])
##                print(all(datetimes_full[dateCount] in datetimes_myvar for datetimes_myvar in datetimes_allvars))  # check if the current datetime from a list of all possible datetimes
#            if all(datetimes_full[dateCount] in datetimes_myvar for datetimes_myvar in datetimes_allvars):  # check if the current datetime from a list of all possible datetimes
#                                                                                                                 # is present for all variables
###
#        datetimes_myvar = datetimes_allvars[varcount]

        fig = plt.figure(figsize=(12, 9))
        node_count = -1
        my_var_cmap = 'Blues'
        my_var_vmin = 0
        if varcount == 0:
            my_var_vmax = 1#2#3#10#25#5
            my_var_levels = [0,0.2,0.4,0.6,0.8,1]#[0,0.4,0.8,1.2,1.6,2]#[0,0.5,1,1.5,2,2.5,3]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
            my_var_tick_levels = [0,0.2,0.4,0.6,0.8,1]#[0,0.4,0.8,1.2,1.6,2]#[0,0.5,1,1.5,2,2.5,3]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
        elif varcount == 1:
            my_var_vmax = 25#20#15#3#10#25#5
            my_var_levels = [0,5,10,15,20,25]#[0,4,8,12,16,20]#[0,3,6,9,12,15]#[0,0.5,1,1.5,2,2.5,3]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
            my_var_tick_levels = [0,5,10,15,20,25]#[0,4,8,12,16,20]#[0,3,6,9,12,15]#[0,0.5,1,1.5,2,2.5,3]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
        my_var_cbarextend = 'max'
        my_var_cbarlabel = 'liquid precipitation [mm]'
        print('datetimes_full = ' + str(datetimes_full))
        for iSOM in range(0, SOMsize[1]):
            for jSOM in range(0, SOMsize[0]):
                node_count = node_count + 1
#                winning_node_indices = winning_nodes[jSOM, iSOM]
                np.set_printoptions(threshold=sys.maxsize)
                winning_node_indices = np.where(predictions == node_count)  # same as above, but for sklearn-som
                print('len(winning_node_indices) = ' + str(len(winning_node_indices)))
                print('np.array(winning_node_indices)[0] = ')# + str(winning_node_indices))
                print(np.array(winning_node_indices)[0])
#                print('np.in1d(winning_node_indices,datetime_6hr_indices_allvars[varcount]) = ' + str(np.in1d(winning_node_indices,datetime_6hr_indices_allvars[varcount])))
#                print('len(np.in1d(winning_node_indices,datetime_6hr_indices_allvars[varcount])) = ' + str(len(np.in1d(winning_node_indices,datetime_6hr_indices_allvars[varcount]))))
##                winning_node_indices = np.nonzero(np.in1d(winning_node_indices,datetime_6hr_indices_allvars[varcount]))[0]  # now only use indices for times that are in 6-hour intervals
#                winning_node_indices = np.array(winning_node_indices)[0][np.in1d(winning_node_indices,datetime_6hr_indices_allvars[varcount])]  # now only use indices for times that are in 6-hour intervals
##                datetime_6hr_indices_mask = np.in1d(winning_node_indices,datetime_6hr_indices_allvars[varcount])  # now only use indices for times that are in 6-hour intervals
##                winning_node_indices = winning_node_indices[datetime_6hr_indices_mask]
#                print('len(winning_node_indices) [after 6-hourly filtering] = ' + str(len(winning_node_indices)))
#                print('winning_node_indices [after 6-hourly filtering] = ' + str(winning_node_indices))
#                print('len(datetimes_myvar) = ' + str(len(datetimes_myvar)))
#                print('datetimes_myvar[winning_node_indices[0]] = ' + str(datetimes_myvar[winning_node_indices[0]]))
#                print('datetimes_myvar[winning_node_indices[1]] = ' + str(datetimes_myvar[winning_node_indices[1]]))
#                print('datetimes_myvar[winning_node_indices[2]] = ' + str(datetimes_myvar[winning_node_indices[2]]))
                if len(winning_node_indices) > 0:
                    print('winning_node_indices[0] = ' + str(winning_node_indices[0]))
                np.set_printoptions(threshold=False)
#                winning_node_indices = np.where((predictions == node_count) & (datetimes_myvar == datetimes_full))  # same as above, but for sklearn-som - with addition of only using data every 6 hours
                # Adjust the number and arrangement of the subplots based on the SOM grid size
                if SOMsize_str == '3x2':
                    my_subplot = 231
                elif SOMsize_str == '4x2':
                    my_subplot = 241
                elif SOMsize_str == '4x3':
                    my_subplot = 341
                elif SOMsize_str == '5x4':
                    my_subplot = 451
                # cart_proj = ccrs.LambertConformal()
                cart_proj = ccrs.PlateCarree()
                ax = plt.subplot(SOMsize[1], SOMsize[0], node_count + 1, projection=cart_proj)
                # print(varSOM.get_weights()[jSOM-1, iSOM-1, :])
#                dataArray_node = dataArrays[varcount][winning_node_indices]
                dataArray_node = dataArrays_orig[varcount][winning_node_indices]
                dataArray_node_map = np.mean(dataArray_node.reshape(-1, mapSize[1], mapSize[0]), axis=0)  # -1 --> value is inferred from the length of the array and remaining dimensions (keeps 1st dim len consistent if 2nd dim len = mapSize[1] * mapSize[0])
#                varSOM_plot = varSOM.get_weights()[jSOM, iSOM, :].reshape(mapSize[1], mapSize[0])
                # print(varSOM_plot)
                # ax_ = ax.contourf(map_lons, map_lats, varSOM, cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
                # ax_ = ax.contourf(map_lons, map_lats, varSOM.get_weights(), cmap=my_var_cmap,
                #                   vmin=my_var_vmin, vmax=my_var_vmax,
                #                   levels=my_var_levels, transform=ccrs.PlateCarree(), extend=my_var_cbarextend)
                # ds = xr.open_dataset("/Users/steven.naegele/Data/MRMS/20210101_20231231/MultiSensor_QPE_01H_Pass2_00.00_20210101-000000.grib2", engine="cfgrib")
                # lats = ds['latitude'][mapedges[2]:mapedges[3]:10, mapedges[0]:mapedges[1]:10]
                # lons = ds['longitude'][mapedges[2]:mapedges[3]:10, mapedges[0]:mapedges[1]:10]
                #ax_ = ax.contourf(map_lats, map_lons, varSOM_plot, cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
#                ax_ = ax.contourf(map_lons, map_lats, np.transpose(varSOM_plot), cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
                ax_ = ax.contourf(map_lons, map_lats, np.transpose(dataArray_node_map), cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
                # ax_ = ax.contourf(lats, lons, varSOM_plot, cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
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
#                ax.title.set_text('Node ' + str(node_count + 1))
                ax.set_title('Node ' + str(node_count + 1), pad=7)
                # ax.outline_patch.set_alpha(0)  # plot outline/gridbox
                # ax.background_patch.set_alpha(0)  # plot background
                # plt.imshow(varSOM_plot, interpolation='none', aspect='auto')
        #cax = plt.axes([0.2, 0.525, 0.625, 0.025])
        cax = plt.axes([0.2, 0.075, 0.625, 0.025])
        cbar = fig.colorbar(ax_, cax=cax, orientation='horizontal', extend=my_var_cbarextend,
                            ticks=my_var_tick_levels, label=my_var_cbarlabel)#, pad=0.05)
        varname_title = myvarname.replace('_',' ')
        sample_filename = '/scratch1/BMC/wrfruc/ejames/climo/mrms/202212/mrms_radaronly_2022123118.nc'
#        filename_first = myfiles[0][0]#[varcount][0]
#        print('filename_first = ' + str(filename_first))
##        date_first = filename_first.split('/')[-1].split('.')[1][3:10+1]
#        date_first = filename_first.split('/')[-1].split('_')[-1][0:7+1]  # e.g. '20221231'
#        print('date_first = ' + str(date_first))
#        filename_last = myfiles[0][-1]#[varcount][-1]
#        print('filename_last = ' + str(filename_last))
##        date_last = filename_last.split('/')[-1].split('.')[1][3:10+1]
#        date_last = filename_last.split('/')[-1].split('_')[-1][0:7+1]
#        print('date_last = ' + str(date_last))
#        datetimes_full = pandas.date_range(date_first,date_last,freq='6h')
#        datetimes_full = datetimes_full.format(formatter=lambda x: x.strftime('%Y%m%d-%H'))
        #plt.suptitle(varname_title + ' -- ' + date_first + '-' + date_last, x=0.5125, y=0.925, fontsize=14)
#        plt.suptitle(varname_title + ' -- (lr=0.25) ' + date_first + '-' + date_last, x=0.5325, y=0.95, fontsize=20)
        plt.suptitle(varname_title + ' -- ' + date_first + '-' + date_last, x=0.5325, y=0.95, fontsize=20)
#        plt.savefig('SOM_' + myvarname + '_01H24Htrain_' + SOMsize_str + '_lr25_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2_clean.png')
        plt.savefig('SOM_' + myvarname + '_01H24Htrain_' + SOMsize_str + '_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2_clean_sklearn_6hr.png')


        #print(varSOM)
#        winning_nodes = varSOM.win_map(dataArray, return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
                                                                        # note that the SOM node indices are [col, row], and I will read through cols first
        # Create a heatmap of number of training cases from each month assigned to each SOM node
        fig = plt.figure(figsize=(50, 10))
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_nums = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        # d = dict(month_nums, month_names)
        d = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
             '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
        year_month_strs_all = []
        seasons_all = []
        sample_filename = '/scratch1/BMC/wrfruc/ejames/climo/mrms/202212/mrms_radaronly_2022123118.nc'
#        for myfile in myfiles[varcount]:
        for myfile in datetimes_ari_allvars[varcount]:
#            myyear = myfile.split('/')[-1].split('.')[1][3:6+1]
#            myyear = myfile.split('/')[-1].split('_')[-1][0:3+1]
            myyear = myfile[0:3+1]
#            mymonthname = d[myfile.split('/')[-1].split('.')[1][7:8+1]]
#            mymonthname = d[myfile.split('/')[-1].split('_')[-1][4:5+1]]
            mymonthname = d[myfile[4:5+1]]
            year_month_str_ = myyear + '-' + mymonthname
            year_month_strs_all.append(year_month_str_)
#            if (mymonthname == 'Jan' or mymonthname == 'Feb') and int(myyear) == years_doubles_allvars[varcount][0]:
            if (mymonthname == 'Jan' or mymonthname == 'Feb') and int(myyear) == int(date_first[0:3+1]):
                seasons_all.append('winter' + ' ' + myyear)
#            elif (mymonthname == 'Jan' or mymonthname == 'Feb') and int(myyear) != years_doubles_allvars[varcount][0]:
            elif (mymonthname == 'Jan' or mymonthname == 'Feb') and int(myyear) != int(date_first[0:3+1]):
                seasons_all.append('winter' + ' ' + str(int(myyear)-1) + '-' + myyear)
            elif mymonthname == 'Mar' or mymonthname == 'Apr' or mymonthname == 'May':
                seasons_all.append('spring' + ' ' + myyear)
            elif mymonthname == 'Jun' or mymonthname == 'Jul' or mymonthname == 'Aug':
                seasons_all.append('summer' + ' ' + myyear)
            elif mymonthname == 'Sep' or mymonthname == 'Oct' or mymonthname == 'Nov':
                seasons_all.append('autumn' + ' ' + myyear)
#            elif mymonthname == 'Dec' and int(myyear) != years_doubles_allvars[varcount][-1]:
            elif mymonthname == 'Dec' and int(myyear) != int(date_last[0:3+1]):
                seasons_all.append('winter' + ' ' + myyear + '-' + str(int(myyear)+1))
#            elif mymonthname == 'Dec' and int(myyear) == years_doubles_allvars[varcount][-1]:
            elif mymonthname == 'Dec' and int(myyear) == int(date_last[0:3+1]):
                seasons_all.append('winter' + ' ' + myyear)
        #print(year_month_strs)
        #print(year_month_strs_all)
        print('len(seasons_all) = ' + str(len(seasons_all)))
        node_count = -1
        #num_months = (datetime(int(date_last[0:3+1]),int(date_last[4:5+1]),int(date_last[6:7+1])).year - datetime(int(date_first[0:3+1]),int(date_first[4:5+1]),int(date_first[6:7+1])).year) * 12 + \
        #             datetime(int(date_last[0:3+1]),int(date_last[4:5+1]),int(date_last[6:7+1])).month - datetime(int(date_first[0:3+1]),int(date_first[4:5+1]),int(date_first[6:7+1])).month
#        print('len(months_doubles_allvars) = ' + str(len(months_doubles_allvars)))
#        print('len(months_doubles_allvars) = ' + str(len(months_doubles_allvars[0])))
#        print(months_doubles_allvars)
#        print('len(year_doubles_allvars) = ' + str(len(years_doubles_allvars)))
#        print('len(year_doubles_allvars) = ' + str(len(years_doubles_allvars[0])))
#        print(years_doubles_allvars)
#        print('len(year_month_strs_allvars) = ' + str(len(year_month_strs_allvars)))
#        print('len(year_month_strs_allvars) = ' + str(len(year_month_strs_allvars[0])))
#        print(year_month_strs_allvars)
#        print('len(year_month_counts_allvars) = ' + str(len(year_month_counts_allvars)))
#        print('len(year_month_counts_allvars) = ' + str(len(year_month_counts_allvars[0])))
#        print(year_month_counts_allvars)

#        num_months = len(year_month_strs_allvars[varcount])
#        year_month_strs = pandas.date_range(date_first[0:3+1]+'-'+date_first[4:5+1]+'-'+date_first[6:7+1],date_last[0:3+1]+'-'+date_last[4:5+1]+'-'+date_last[6:7+1] - timedelta(months=1),freq='MS')
#        year_month_strs = pandas.date_range(date_first[0:3+1]+'-'+date_first[4:5+1]+'-'+date_first[6:7+1],date_last[0:3+1]+'-'+date_last[4:5+1]+'-'+date_last[6:7+1],freq='MS')
        year_month_strs = pandas.date_range(date_first[0:3+1]+'-'+date_first[4:5+1]+'-01',date_last[0:3+1]+'-'+date_last[4:5+1]+'-01',freq='MS')
        print('len(year_month_strs) = ' + str(len(year_month_strs)))
#        year_month_strs.format(formatter=lambda x: x.strftime('%Y%m'))
        year_month_strs = [x.strftime('%Y%m%d') for x in year_month_strs]
#        [x = x[0:3+1]+x[5:6+1] for x in year_month_strs]
#        [x.str.replace('-','') for x in year_month_strs]
#        [x.astype(str) for x in year_month_strs]
        print(type(year_month_strs[0]))
        num_months = len(year_month_strs)
#        num_months = len(datetimes_full_yearmon)
        monthly_sums = np.zeros((np.prod(SOMsize), num_months))
        seasons_list = []
#        for myyearmonth in year_month_strs_allvars[varcount]:
        for myyearmonth in year_month_strs:
#        for myyearmonth in datetimes_full_yearmon:
            print('myyearmonth = ' + str(myyearmonth))  # myyearmonth = 2015-03-01 00:00:00
#            myyear = myyearmonth.strftime('%Y%m%d')[0:3+1]
            myyear = myyearmonth[0:3+1]
#            mymonthname = myyearmonth[5:7+1]
#            print('myyearmonth.strftime("%Y%m%d") = ' + str(myyearmonth.strftime('%Y%m%d')))
#            print('myyearmonth.strftime("%Y%m%d")[4:5+1] = ' + str(myyearmonth.strftime('%Y%m%d')[4:5+1]))
            print('myyearmonth = ' + str(myyearmonth))
            print('myyearmonth[4:5+1] = ' + str(myyearmonth[4:5+1]))
#            mymonthname = d[myyearmonth.strftime('%Y%m%d')[4:5+1]]
            mymonthname = d[myyearmonth[4:5+1]]
#            if myyearmonth == myyearmonth[0]:
#                print(mymonthname)
#            if (mymonthname == 'Jan' or mymonthname == 'Feb') and int(myyear) == years_doubles_allvars[varcount][0]:
            if (mymonthname == 'Jan' or mymonthname == 'Feb') and int(myyear) == int(date_first[0:3+1]):
                if 'winter' + ' ' + myyear not in seasons_list:
                    print('winter' + ' ' + myyear)
                    seasons_list.append('winter' + ' ' + myyear)
#            elif (mymonthname == 'Jan' or mymonthname == 'Feb') and int(myyear) != years_doubles_allvars[varcount][0]:
            elif (mymonthname == 'Jan' or mymonthname == 'Feb') and int(myyear) != int(date_first[0:3+1]):
                if 'winter' + ' ' + str(int(myyear)-1) + '-' + myyear not in seasons_list:
                    print('winter' + ' ' + str(int(myyear)-1) + '-' + myyear)
                    seasons_list.append('winter' + ' ' + str(int(myyear)-1) + '-' + myyear)
            elif mymonthname == 'Mar' or mymonthname == 'Apr' or mymonthname == 'May':
                if 'spring' + ' ' + myyear not in seasons_list:
                    print('spring' + ' ' + myyear)
                    seasons_list.append('spring' + ' ' + myyear)
            elif mymonthname == 'Jun' or mymonthname == 'Jul' or mymonthname == 'Aug':
                if 'summer' + ' ' + myyear not in seasons_list:
                    print('summer' + ' ' + myyear)
                    seasons_list.append('summer' + ' ' + myyear)
            elif mymonthname == 'Sep' or mymonthname == 'Oct' or mymonthname == 'Nov':
                if 'autumn' + ' ' + myyear not in seasons_list:
                    print('autumn' + ' ' + myyear)
                    seasons_list.append('autumn' + ' ' + myyear)
#            elif mymonthname == 'Dec' and int(myyear) != years_doubles_allvars[varcount][-1]:
            elif mymonthname == 'Dec' and int(myyear) != int(date_last[0:3+1]):
                if 'winter' + ' ' + myyear + '-' + str(int(myyear)+1) not in seasons_list:
                    print('winter' + ' ' + myyear + '-' + str(int(myyear)+1))
                    seasons_list.append('winter' + ' ' + myyear + '-' + str(int(myyear)+1))
#            elif mymonthname == 'Dec' and int(myyear) == years_doubles_allvars[varcount][-1]:
            elif mymonthname == 'Dec' and int(myyear) == int(date_last[0:3+1]):
                if 'winter' + ' ' + myyear not in seasons_list:
                    print('winter' + ' ' + myyear)
                    seasons_list.append('winter' + ' ' + myyear)
        num_seasons = len(seasons_list)
        print('num_seasons = ' + str(num_seasons))
        print(seasons_list)
        seasonly_sums = np.zeros((np.prod(SOMsize), num_seasons))
        print('datetimes_full_yearmon = ' + str(datetimes_full_yearmon))
        print('len(datetimes_full_yearmon) = ' + str(len(datetimes_full_yearmon)))
        for iSOM in range(0, SOMsize[1]):  # row
            for jSOM in range(0, SOMsize[0]):  #col
#                winning_node_indices = winning_nodes[jSOM, iSOM]
                print(str(jSOM) + ', ' + str(iSOM))
                node_count = node_count + 1
                print('len(predictions) = ' + str(len(predictions)))
                winning_node_indices = np.where(predictions == node_count)  # same as above, but for sklearn-som
                print('len(winning_node_indices) [again] = ' + str(len(winning_node_indices)))
                print('np.array(winning_node_indices)[0] [again] = ')# + str(winning_node_indices))
                print(np.array(winning_node_indices)[0])
#                winning_node_indices = np.nonzero(np.in1d(winning_node_indices,datetime_6hr_indices_allvars[varcount]))[0]  # now only use indices for times that are in 6-hour intervals
#                winning_node_indices = np.array(winning_node_indices)[0][np.in1d(winning_node_indices,datetime_6hr_indices_allvars[varcount])]  # now only use indices for times that are in 6-hour intervals
#                datetime_6hr_indices_mask = np.in1d(winning_node_indices,datetime_6hr_indices_allvars[varcount])  # now only use indices for times that are in 6-hour intervals
#                winning_node_indices = winning_node_indices[datetime_6hr_indices_mask]
#                print('len(winning_node_indices) = ' + str(len(winning_node_indices)))
#                print('winning_node_indices = ' + str(winning_node_indices))
#                if len(winning_node_indices) > 0:
#                    print('winning_node_indices[0] = ' + str(winning_node_indices[0]))
                if jSOM == 2 and iSOM == 4:
                    print('winning_node_indices[0][0] = ' + str(winning_node_indices[0][1]))
                #if not 'node_numbers' in locals():  # if the dictionary for SOM node 2D indices and node count does not already exist, create it
                #    node_numbers = {(jSOM, iSOM): node_count}
                #else:  # if the the dictionary does exist, append new numbers to it
                #    node_numbers[(jSOM, iSOM)] = node_count
                #for imonth in range(0, num_months):
                year_month_str_count = -1
#                for year_month_str in year_month_strs_allvars[varcount]:
                print('year_month_strs = ' + str(year_month_strs))
                print('len(year_month_strs) = ' + str(len(year_month_strs)))
                for year_month_str in year_month_strs:
#                for year_month_str in datetimes_full_yearmon:
#                    sample_filename = 'MultiSensor_QPE_01H_Pass2_00.00_20210101-000000.nc'
                    sample_filename = '/scratch1/BMC/wrfruc/ejames/climo/mrms/202212/mrms_radaronly_2022123118.nc'
                    #year_mon_str = myfile.split('/')[-1].split('.')[1][3:6+1] + '-' + myfile.split('/')[-1].split('.')[1][7:8+1]
                    #year_month_str = year_month_str[0:3+1] + year_month_str[5:6+1]  # get rid of the dash I inserted when originally making this variable, which I now realize was a mistake
                    #winning_node_indices_month = np.where((months_doubles[winning_node_indices] == imonth) & years_doubles[winning_node_indices] == )
                    #winning_node_indices_month = np.where(myfiles[np.array(winning_node_indices)].split('/')[-1].split('.')[1][3:8+1] == year_month_str)
                    #indices_month = np.where(myfiles.split('/')[-1].split('.')[1][3:8+1] == year_month_str)
#                    indices_month = np.where(np.array(year_month_strs_all) == year_month_str.strftime('%Y%m%d')[0:3+1]+'-'+d[year_month_str.strftime('%Y%m%d')[4:5+1]])
                    print('year_month_str = ' + str(year_month_str))
#                    indices_month = np.where(datetimes_full_yearmon == year_month_str.strftime('%Y%m%d')[0:3+1]+'-'+d[year_month_str.strftime('%Y%m%d')[4:5+1]])
                    print('datetimes_full_yearmon[0] = ' + datetimes_full_yearmon[0])
                    print('str(year_month_str)[0:6+1] = ' + str(year_month_str)[0:5+1])#[0:6+1])
#                    indices_month = np.where(np.array(datetimes_full_yearmon) == str(year_month_str)[0:5+1])#[0:6+1])
#                    print('year_month_str = ' + str(year_month_str.strftime('%Y%m%d')[0:3+1]+'-'+d[year_month_str.strftime('%Y%m%d')[4:5+1]]))  # e.g. 2022-Jan
                    print('year_month_str = ' + str(year_month_str[0:3+1]+'-'+d[year_month_str[4:5+1]]))  # e.g. 2022-Jan
#                    print('year_month_strs_all = ' + str(year_month_strs_all))
                    #print('year_month_str = ' + year_month_str)
                    ##print('length of winning_node_indices_month = ' + str(len(winning_node_indices_month)))
#                    print('indices_month = ' + str(indices_month))
#                    print('indices_month[0] = ' + str(indices_month[0]))
 #                   print('np.array(indices_month) = ' + str(np.array(indices_month)))
 #                   print('np.array(indices_month)[0] = ' + str(np.array(indices_month)[0]))
                    if len(winning_node_indices) > 0:
#                        mymask = np.isin(indices_month[0], winning_node_indices[0])  # if an index in winning_node_indices is in indices_month, assign the corresponding element in a masked array of length len(indices_month) to True
#                        print('mymask = ' + str(mymask))
#                        winning_node_indices_month = indices_month[0][mymask]  # only use the winning node indices for the current month
#                        print('winning_node_indices_month = ' + str(winning_node_indices_month))
##                        winning_node_indices_month = indices_month[mymask]  # only use the winning node indices for the current monthi
#                        print('shared_datetimes = ' + str(shared_datetimes))
#                        print('type(datetimes_full_yearmon) = ' + str(type(datetimes_full_yearmon)))
#                        print('type(winning_node_indices_month) = ' + str(type(winning_node_indices_month)))
#                        print('len(datetimes_full_yearmon) = ' + str(len(datetimes_full_yearmon)))
#                        print('len(winning_node_indices_month) = ' + str(len(winning_node_indices_month)))
#                        print('len(predictions) = ' + str(len(predictions)))
#                        print('np.array(datetimes_full_yearmon)[np.array(winning_node_indices_month)] = ' + str(np.array(datetimes_full_yearmon)[np.array(winning_node_indices_month)]))
#                        mymask2 = np.isin(np.array(datetimes_full_yearmon)[np.array(winning_node_indices_month)], shared_datetimes[:][0:5+1])  # now need to only use the winning node indices for the current month that are also shared among all variables
#                        winning_node_indices_month = winning_node_indices_month[mymask2]   # now need to only use the winning node indices for the current month that are also shared among all variables

                        #mymask = np.isin(np.array(datetimes_full_yearmon), shared_datetimes[:][0:5+1]) # if a year-month is shared among the variables as one that was trained upon, assign the corresponding element in a masked array to True
                        #datetimes_full_yearmon_shared = datetimes_full_yearmon[0][mymask]
                        #datetimes_full_yearmon_shared_month = np.where((np.array(datetimes_full_yearmon) == shared_datetimes[:][0:5+1]) and (np.array(datetimes_full_yearmon) == str(year_month_str)[0:5+1]))
#                        print('datetimes_full_yearmon = ' + str(datetimes_full_yearmon))  # 11488 long
#                        print('shared_datetimes[:][0:5+1] = ' + str(shared_datetimes[:][0:5+1]))
                        shared_datetimes_yearmon = [x[0:5+1] for x in shared_datetimes]  # 11170 long, when it should be 9934(?)
#                        print('shared_datetimes_yearmon = ' + str(shared_datetimes_yearmon))
#                        print('len(datetimes_full_yearmon) = ' + str(len(datetimes_full_yearmon)))
#                        print('len(shared_datetimes_yearmon) = ' + str(len(shared_datetimes_yearmon)))
#                        print('datetimes_full = ')
#                        print(datetimes_full)
#                        print('shared_datetimes = ')
#                        print(shared_datetimes)
#                        print('type(datetimes_full) = ' + str(type(datetimes_full)))
#                        print('type(datetimes_full[0]) = ' + str(type(datetimes_full[0])))
#                        print('type(shared_datetimes) = ' + str(type(shared_datetimes)))
#                        print('type(shared_datetimes[0]) = ' + str(type(shared_datetimes[0])))
##                        datetimes_full_yearmon_shared_indices = np.where(datetimes_full_yearmon == shared_datetimes_yearmon)  # find the indices where the ARI threshold is exceed for at least one variable among all 6-hour datetimes
                        datetimes_full_yearmon_shared_indices = np.nonzero(np.in1d(datetimes_full, shared_datetimes))[0]  # find the indices where the ARI threshold is exceed for at least one variable among all 6-hour datetimes
#                        print(str(jSOM) + ', ' + str(iSOM))
#                        print('datetimes_full_yearmon_shared_indices = ' + str(datetimes_full_yearmon_shared_indices))
#                        print('len(datetimes_full_yearmon_shared_indices) = ' + str(len(datetimes_full_yearmon_shared_indices)))
                        datetimes_full_yearmon_shared_indices = [int(x) for x in datetimes_full_yearmon_shared_indices]
#                        print('datetimes_full_yearmon_shared_indices [int] = ' + str(datetimes_full_yearmon_shared_indices))
#                        print('len(datetimes_full_yearmon_shared_indices) [int] = ' + str(len(datetimes_full_yearmon_shared_indices)))
#                        print('np.array(datetimes_full_yearmon)[0] = ' + str(np.array(datetimes_full_yearmon)))
#                        print('np.array(datetimes_full_yearmon_shared_indices)[0]) [int] = ' + str(np.array(datetimes_full_yearmon_shared_indices)))
#                        print('len(np.array(datetimes_full_yearmon_shared_indices)[0]) [int] = ' + str(len(np.array(datetimes_full_yearmon_shared_indices))))
                        datetimes_full_yearmon_shared = np.array(datetimes_full_yearmon)[np.array(datetimes_full_yearmon_shared_indices)]  # array of 6-hourly datetimes that have at least one variable exceed the ARI threshold
                        datetimes_full_shared = np.array(datetimes_full)[np.array(datetimes_full_yearmon_shared_indices)]  # array of 6-hourly datetimes that have at least one variable exceed the ARI threshold
#                        print('datetimes_full_yearmon_shared = ' + str(datetimes_full_yearmon_shared))
                        #mymask = np.isin(indices_month[0], winning_node_indices[0])  # if an index in winning_node_indices is in indices_month, assign the corresponding element in a masked array of length len(indices_month) to True
                        indices_month = np.where(datetimes_full_yearmon_shared == str(year_month_str)[0:5+1])  # in the array of ARI-filtered 6-hourly datetimes, find indices that only correspond to the current year and month
#                        print('indices_month = ' + str(indices_month))
                        mymask = np.isin(indices_month[0], winning_node_indices[0])  # create a mask that will highlight the indices in indices_month that are also clustered to the current SOM node
#                        print('mymask = ' + str(mymask))
                        winning_node_indices_month = indices_month[0][mymask]
#                        print('winning_node_indices_month = ' + str(winning_node_indices_month))
                        if year_month_str[0:3] == '202':
                            print('len(datetimes_full) = ' + str(len(datetimes_full)) + ', len(datetimes_full_yearmon) = ' + str(len(datetimes_full_yearmon)) + ', len(shared_datetimes) = ' + str(len(datetimes_full)))  # 11488, 11488
                            print('datetimes_full_shared[:10] = ' + str(datetimes_full_shared[:10]))
                            print('datetimes_full_shared[-10:] = ' + str(datetimes_full_shared[-10:]))
                            print('[year_month_str = ' + year_month_str + '] datetimes_full_yearmon_shared_indices = ' + str(datetimes_full_yearmon_shared_indices))  # [year_month_str = 20221201] datetimes_full_yearmon_shared_indices = [3, 4, 7, ..., 11478, 11482, 11486]
                            print('datetimes_full[10026, 10027, 10030, 10034] = ' + datetimes_full[10026] + ', ' + datetimes_full[10027] + ', ' + datetimes_full[10030] + ', ' + datetimes_full[10034])  # 20211231-12, 20211231-18, 20220101-12, 20220102-12
                            print('len(datetimes_full_shared) = ' + str(len(datetimes_full_shared)) + ', len(datetimes_full_yearmon_shared) = ' + str(len(datetimes_full_yearmon_shared)))
                            print('len(indices_month) = ' + str(len(indices_month[0])) + ', len(datetimes_full_yearmon_shared) = ' + str(len(datetimes_full_yearmon_shared)))  # ~110 for 2015-2021 & ~30 for 2022, 9626
                            print('[year_month_str = ' + year_month_str + '] indices_month = ' + str(indices_month))  # [year_month_str = 20221201] indices_month = (array([9595, 9596, 9597, ..., 9623, 9624, 9625]),)
                            print('[year_month_str = ' + year_month_str + '] winning_node_indices_month = ' + str(winning_node_indices_month))  # [year_month_str = 20221201] winning_node_indices_month = [9597 9603 9604 ... 9623 9624 9625]
                    else:
                        winning_node_indices_month = []
#                    print('winning_node_indices_month [after shared_datetimes filtering] = ' + str(winning_node_indices_month))
#                    print('len(winning_node_indices_month) [after shared_datetimes filtering] = ' + str(len(winning_node_indices_month)))
                    #print(winning_node_indices_month)
                    year_month_str_count = year_month_str_count + 1
                    monthly_sums[node_count, year_month_str_count] = monthly_sums[node_count, year_month_str_count] + len(winning_node_indices_month)
                season_str_count = -1
                for season_str in seasons_list:
                    print(str(jSOM) + ', ' + str(iSOM) + ', ' + str(season_str))
                    indices_season = np.where(np.array(seasons_all) == season_str)
                    print('indices_season = ' + str(indices_season))
                    print('len(indices_season) = ' + str(len(indices_season)))
#                    mymask2 = np.isin(indices_season, winning_node_indices)
                    mymask2 = np.isin(indices_season[0], winning_node_indices[0])
                    print('mymask2 = ' + str(mymask2))
                    print('len(mymask2) = ' + str(len(mymask2)))
#                    winning_node_indices_season = np.array(indices_season)[mymask2]
                    winning_node_indices_season = indices_season[0][mymask2]
                    print('winning_node_indices_season = ' + str(winning_node_indices_season))
                    print('len(winning_node_indices_season) = ' + str(len(winning_node_indices_season)))
                    season_str_count = season_str_count + 1
                    seasonly_sums[node_count, season_str_count] = seasonly_sums[node_count, season_str_count] + len(winning_node_indices_season)
        title_fontsize = 28
        axis_fontsize = 18
        annot_fontsize = 14
        np.set_printoptions(threshold=sys.maxsize)
        print('monthly_sums = ' + str(monthly_sums))
#        ax = sns.heatmap(monthly_sums, annot=True, fmt='g', xticklabels=year_month_strs_allvars[varcount], yticklabels=np.arange(1, np.prod(SOMsize)+1), annot_kws={'size': annot_fontsize}, cbar_kws={'pad': 0.01})
        ax = sns.heatmap(monthly_sums, annot=True, fmt='g', xticklabels=year_month_strs, yticklabels=np.arange(1, np.prod(SOMsize)+1), annot_kws={'size': annot_fontsize}, cbar_kws={'pad': 0.01})
        #ax.set_xlabel('month', fontsize=axis_fontsize)
        ax.set_ylabel('SOM node', fontsize=axis_fontsize)
        plt.yticks(rotation=0, fontsize=axis_fontsize)
        plt.xticks(rotation=45, ha='right', fontsize=axis_fontsize)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=axis_fontsize)
        ax.figure.subplots_adjust(left = 0.3)
#        plt.title('Maps per SOM node per month -- ' + varname_title + ' (lr=0.25)', fontsize=title_fontsize)
        plt.title('Maps per SOM node per month -- ' + varname_title, fontsize=title_fontsize)
#        plt.savefig('SOM_heatmap_' + myvarname + '_01H24Htrain_' + SOMsize_str + '_lr25_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2_clean.png', bbox_inches='tight', pad_inches=0.01)
        plt.savefig('SOM_heatmap_' + myvarname + '_01H24Htrain_' + SOMsize_str + '_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2_clean_sklearn_6hr.png', bbox_inches='tight', pad_inches=0.01)

        fig = plt.figure(figsize=(50, 10))
        print('seasonly_sums = ' + str(seasonly_sums))
        np.set_printoptions(threshold=False)
        ax = sns.heatmap(seasonly_sums, annot=True, fmt='g', xticklabels=seasons_list, yticklabels=np.arange(1, np.prod(SOMsize)+1), annot_kws={'size': annot_fontsize}, cbar_kws={'pad': 0.01})
        ax.set_ylabel('SOM node', fontsize=axis_fontsize)
        plt.yticks(rotation=0, fontsize=axis_fontsize)
        plt.xticks(rotation=45, ha='right', fontsize=axis_fontsize)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=axis_fontsize)
        ax.figure.subplots_adjust(left = 0.3)
#        plt.title('Maps per SOM node per season -- ' + varname_title + ' (lr=0.25)', fontsize=title_fontsize)
        plt.title('Maps per SOM node per season -- ' + varname_title, fontsize=title_fontsize)
#        plt.savefig('SOM_heatmap_season_' + myvarname + '_01H24Htrain_' + SOMsize_str + '_lr25_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2_clean.png', bbox_inches='tight', pad_inches=0.01)
        plt.savefig('SOM_heatmap_season_' + myvarname + '_01H24Htrain_' + SOMsize_str + '_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2_clean_sklearn_6hr.png', bbox_inches='tight', pad_inches=0.01)

        #win_positions = []
        #for myfile in myfiles:
            #win_position = varSOM.winner(myfile)  # get the winning SOM node for each output time
            #print(win_position)
            #win_positions.append(win_position)


#        print(dataArrays[varcount].shape)
        print(dataArrays_orig[varcount].shape)
        fig = plt.figure(figsize=(12, 9))
        node_count = -1
        my_var_cmap = 'Blues'
        my_var_vmin = 0
        if varcount == 0:
            my_var_vmax = 5#15#5
            my_var_levels = [0,1,2,3,4,5]#[0,3,6,9,12,15]#[0,1,2,3,4,5]
            my_var_tick_levels = [0,1,2,3,4,5]#[0,3,6,9,12,15]#[0,1,2,3,4,5]
        elif varcount == 1:
            my_var_vmax = 25#20#15#5
            my_var_levels = [0,5,10,15,20,25]#[0,4,8,12,16,20]#[0,3,6,9,12,15]#[0,1,2,3,4,5]
            my_var_tick_levels = [0,5,10,15,20,25]#[0,4,8,12,16,20]#[0,3,6,9,12,15]#[0,1,2,3,4,5]
        my_var_cbarextend = 'max'
        my_var_cbarlabel = 'liquid precipitation [mm]'
        for iSOM in range(0, SOMsize[1]):  # row
            for jSOM in range(0, SOMsize[0]):  # col
#                winning_node_indices = winning_nodes[jSOM, iSOM]
                node_count = node_count + 1
                winning_node_indices = np.where(predictions == node_count)  # same as above, but for sklearn-som
#                winning_node_indices = np.array(winning_node_indices)[0][np.in1d(winning_node_indices,datetime_6hr_indices_allvars[varcount])]  # now only use indices for times that are in 6-hour intervals
#                dataArray_node = dataArray[winning_node_indices]
#                dataArray_node = dataArrays[varcount][winning_node_indices]
                dataArray_node = dataArrays_orig[varcount][winning_node_indices]
                print(dataArray_node.shape)
                dataArray_node_map = dataArray_node.reshape(-1, mapSize[1], mapSize[0])  # -1 --> value is inferred from the length of the array and remaining dimensions (keeps 1st dim len consistent if 2nd dim len = mapSize[1] * mapSize[0])
                print(dataArray_node_map.shape)
                dataArray_node_map_std = np.std(dataArray_node_map, axis=0)  # standard deviation across time for each grid point
                print(dataArray_node_map_std)
                # Adjust the number and arrangement of the subplots based on the SOM grid size
                if SOMsize_str == '3x2':
                    my_subplot = 231
                elif SOMsize_str == '4x2':
                    my_subplot = 241
                elif SOMsize_str == '4x3':
                    my_subplot = 341
                elif SOMsize_str == '5x4':
                    my_subplot = 451
                # cart_proj = ccrs.LambertConformal()
                cart_proj = ccrs.PlateCarree()
                ax = plt.subplot(SOMsize[1], SOMsize[0], node_count + 1, projection=cart_proj)
                ax_ = ax.contourf(map_lons, map_lats, np.transpose(dataArray_node_map_std), cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
                # ax_ = ax.contourf(lats, lons, varSOM_plot, cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
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
#                ax.title.set_text('Node ' + str(node_count + 1))
                ax.set_title('Node ' + str(node_count + 1), pad=7)
                # ax.outline_patch.set_alpha(0)  # plot outline/gridbox
                # ax.background_patch.set_alpha(0)  # plot background
                # plt.imshow(varSOM_plot, interpolation='none', aspect='auto')
        #cax = plt.axes([0.2, 0.525, 0.625, 0.025])
        cax = plt.axes([0.2, 0.075, 0.625, 0.025])
        cbar = fig.colorbar(ax_, cax=cax, orientation='horizontal', extend=my_var_cbarextend,
                            ticks=my_var_tick_levels, label=my_var_cbarlabel)#, pad=0.05)
        varname_title = myvarname.replace('_',' ')
        filename_first = myfiles[varcount][0]
#        date_first = filename_first.split('/')[-1].split('.')[1][3:10+1]
        filename_last = myfiles[varcount][-1]
#        date_last = filename_last.split('/')[-1].split('.')[1][3:10+1]
        #plt.suptitle(r'$\sigma$( ' + varname_title + ' ) -- ' + date_first + '-' + date_last, x=0.5125, y=0.9125, fontsize=14)
#        plt.suptitle(r'$\sigma$( ' + varname_title + ' ) (lr=0.25) -- ' + date_first + '-' + date_last, x=0.5125, y=0.95, fontsize=20)
        plt.suptitle(r'$\sigma$( ' + varname_title + ' ) -- ' + date_first + '-' + date_last, x=0.5125, y=0.95, fontsize=20)
#        plt.savefig('SOM_' + myvarname + '_std_01H24Htrain_' + SOMsize_str + '_lr25_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2_clean.png')
        plt.savefig('SOM_' + myvarname + '_std_01H24Htrain_' + SOMsize_str + '_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2_clean_sklearn_6hr.png')


        #varSOM2 = MiniSom(SOMsize[0], SOMsize[1], np.prod(mapSize), sigma=1.5, learning_rate=0.75)
        ## varSOM2.pca_weights_init(dataArray)
        #max_iter = 1000
        #q_error = []
        #t_error = []
        #
        #for i in range(max_iter):
        #    if i % 100 == 0:
        #        print('i = ' + str(i))
        #    rand_i = np.random.randint(len(dataArray))
        #    varSOM2.update(dataArray[rand_i], varSOM2.winner(dataArray[rand_i]), i, max_iter)
        #    q_error.append(varSOM2.quantization_error(dataArray))
        #    t_error.append(varSOM2.topographic_error(dataArray))
        #
        #plt.figure(figsize=(6, 6))
        #plt.plot(np.arange(max_iter), q_error, label='quantization error')
        #plt.plot(np.arange(max_iter), t_error, label='topographic error')
        #plt.ylabel('error')
        #plt.xlabel('iteration index')
        #plt.legend()
        #plt.title(varname_title + ' ' + myregion + ' ' + SOMsize_str + ' SOM error during training')
        #plt.savefig('SOM_' + varname + '_q_t_error_' + SOMsize_str + '_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2.png')
        #print('quantization error = ' + str(varSOM.quantization_error(dataArray)))
        #print('topographic error = ' + str(varSOM.topographic_error(dataArray)))

