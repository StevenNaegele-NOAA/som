# Import necessary modules
import glob
import sys
import numpy as np
from mysom2 import mysom2  # my own function for training the SOM
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import xarray as xr
import pickle
from datetime import datetime
import seaborn as sns
from minisom import MiniSom

myregion_plot = str(sys.argv[1])
print('My region =', myregion_plot)

mySOMsizes = [[5,5]]#[[2,2], [3,2], [3,3], [4,3], [4,4], [5,4], [5,5]]

SOMcount = -1

for mySOMsize in mySOMsizes:

    SOMcount = SOMcount + 1

    SOMsize_str = str(mySOMsize[0]) + 'x' + str(mySOMsize[1])

    print('My SOMsize = ', SOMsize_str)

##with open('SOM_3x2_MultiSensor_QPE_01H_Pass2_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
##with open('SOM_4x3_MultiSensor_QPE_01H_Pass2_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
##    with open('SOM_' + SOMsize_str + '_learningrate25_MultiSensor_QPE_01H_24H_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
#    with open('SOM_' + SOMsize_str + '_thresh_percentile_99th_MultiSensor_QPE_01H_24H_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
##    varSOM, dataArray, varSOM_nonzero, dataArray_nonzero, map_lats, map_lons, months_doubles, years_doubles, \
##    varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles, \
#        varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars, \
#        year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapSize, myregion = pickle.load(f)
##    year_month_strs, year_month_counts, mapedges, myfiles, varname, SOMsize, mapSize, myregion = pickle.load(f)
    ari = '1yr'
    precip_duration1 = '1hr'
    precip_duration2 = '24hr'
    varname_pickle = 'radaronly_QPE_01H_24H'  # variable name for the pickle file where output from this script will be saved
    #with open('SOM_' + SOMsize_str + '_learningrate25_MultiSensor_QPE_01H_24H_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
    #with open('SOM_' + SOMsize_str + '_thresh_percentile_99th_MultiSensor_QPE_01H_24H_' + myregion_plot + '_vars_batch.pkl', 'rb') as f:
    with open('SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' +  myregion_plot + '_vars_batch_part2.pkl', 'rb') as f:
        varSOM, dataArrays_orig, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars, \
        year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapedges, myregion = pickle.load(f)

    maptedge, mapbedge, mapledge, mapredge = mapedges

    winning_nodes = varSOM.win_map(dataArray_train, return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
                                                                          # note that the SOM node indices are [col, row], and I will read through cols first

    varcount = -1

    for myvarname in varnames:

        print('My varname = ', myvarname)

        varcount = varcount + 1

        fig = plt.figure(figsize=(12, 9))
        node_count = -1
        my_var_cmap = 'Blues'
        my_var_vmin = 0
        if varcount == 0:
            my_var_vmax = 2#3#10#25#5
            my_var_levels = [0,0.4,0.8,1.2,1.6,2]#[0,0.5,1,1.5,2,2.5,3]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
            my_var_tick_levels = [0,0.4,0.8,1.2,1.6,2]#[0,0.5,1,1.5,2,2.5,3]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
        elif varcount == 1:
            my_var_vmax = 25#20#15#3#10#25#5
            my_var_levels = [0,5,10,15,20,25]#[0,4,8,12,16,20]#[0,3,6,9,12,15]#[0,0.5,1,1.5,2,2.5,3]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
            my_var_tick_levels = [0,5,10,15,20,25]#[0,4,8,12,16,20]#[0,3,6,9,12,15]#[0,0.5,1,1.5,2,2.5,3]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
        my_var_cbarextend = 'max'
        my_var_cbarlabel = 'liquid precipitation [mm]'
        for iSOM in range(0, SOMsize[1]):
            for jSOM in range(0, SOMsize[0]):
                node_count = node_count + 1
                winning_node_indices = winning_nodes[jSOM, iSOM]
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
                dataArray_node = dataArrays[varcount][winning_node_indices]
#                dataArray_node_map = np.mean(dataArray_node.reshape(-1, mapSize[1], mapSize[0]), axis=0)  # -1 --> value is inferred from the length of the array and remaining dimensions (keeps 1st dim len consistent if 2nd dim len = mapSize[1] * mapSize[0])
                dataArray_node_map = np.nanmean(dataArray_node.reshape(-1, int((mapredge-mapledge)/10), int((mapbedge-maptedge)/10)), axis=0)  # -1 --> value is inferred from the length of the array and remaining dimensions (keeps 1st dim len consistent if 2nd dim len = mapSize[1] * mapSize[0])
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
                ax.title.set_text('Node ' + str(node_count + 1))
                # ax.outline_patch.set_alpha(0)  # plot outline/gridbox
                # ax.background_patch.set_alpha(0)  # plot background
                # plt.imshow(varSOM_plot, interpolation='none', aspect='auto')
        #cax = plt.axes([0.2, 0.525, 0.625, 0.025])
        cax = plt.axes([0.2, 0.075, 0.625, 0.025])
        cbar = fig.colorbar(ax_, cax=cax, orientation='horizontal', extend=my_var_cbarextend,
                            ticks=my_var_tick_levels, label=my_var_cbarlabel)#, pad=0.05)
        varname_title = myvarname.replace('_',' ')
        filename_first = myfiles[varcount][0]
        date_first = filename_first.split('/')[-1].split('.')[1][3:10+1]
        filename_last = myfiles[varcount][-1]
        date_last = filename_last.split('/')[-1].split('.')[1][3:10+1]
        #plt.suptitle(varname_title + ' -- ' + date_first + '-' + date_last, x=0.5125, y=0.925, fontsize=14)
        plt.suptitle(varname_title + ' (ari' + ari') -- ' + date_first + '-' + date_last, x=0.5325, y=0.95, fontsize=14)
#        plt.savefig('SOM_' + myvarname + '_01H24Htrain_' + SOMsize_str + '_thresh_percentile_99th_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2.png')
        plt.savefig('SOM_' + myvarname + '_01H24Htrain_' + SOMsize_str + '_thresh_ari' + ari + '_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2.png')


        #print(varSOM)
#        winning_nodes = varSOM.win_map(dataArray, return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
                                                                        # note that the SOM node indices are [col, row], and I will read through cols first
        # Create a heatmap of number of training cases from each month assigned to each SOM node
        fig = plt.figure(figsize=(27, 9))
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_nums = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        # d = dict(month_nums, month_names)
        d = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
             '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
        year_month_strs_all = []
        seasons_all = []
        for myfile in myfiles[varcount]:
            myyear = myfile.split('/')[-1].split('.')[1][3:6+1]
            mymonthname = d[myfile.split('/')[-1].split('.')[1][7:8+1]]
            year_month_str_ = myyear + '-' + mymonthname
            year_month_strs_all.append(year_month_str_)
            if (mymonthname == 'Jan' or mymonthname == 'Feb') and int(myyear) == years_doubles_allvars[varcount][0]:
                seasons_all.append('winter' + ' ' + myyear)
            elif (mymonthname == 'Jan' or mymonthname == 'Feb') and int(myyear) != years_doubles_allvars[varcount][0]:
                seasons_all.append('winter' + ' ' + str(int(myyear)-1) + '-' + myyear)
            elif mymonthname == 'Mar' or mymonthname == 'Apr' or mymonthname == 'May':
                seasons_all.append('spring' + ' ' + myyear)
            elif mymonthname == 'Jun' or mymonthname == 'Jul' or mymonthname == 'Aug':
                seasons_all.append('summer' + ' ' + myyear)
            elif mymonthname == 'Sep' or mymonthname == 'Oct' or mymonthname == 'Nov':
                seasons_all.append('autumn' + ' ' + myyear)
            elif mymonthname == 'Dec' and int(myyear) != years_doubles_allvars[varcount][-1]:
                seasons_all.append('winter' + ' ' + myyear + '-' + str(int(myyear)+1))
            elif mymonthname == 'Dec' and int(myyear) == years_doubles_allvars[varcount][-1]:
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
        num_months = len(year_month_strs_allvars[varcount])
        monthly_sums = np.zeros((np.prod(SOMsize), num_months))
        seasons_list = []
        for myyearmonth in year_month_strs_allvars[varcount]:
            myyear = myyearmonth[0:3+1]
            mymonthname = myyearmonth[5:7+1]
            if myyearmonth == myyearmonth[0]:
                print(mymonthname)
            if (mymonthname == 'Jan' or mymonthname == 'Feb') and int(myyear) == years_doubles_allvars[varcount][0]:
                if 'winter' + ' ' + myyear not in seasons_list:
                    print('winter' + ' ' + myyear)
                    seasons_list.append('winter' + ' ' + myyear)
            elif (mymonthname == 'Jan' or mymonthname == 'Feb') and int(myyear) != years_doubles_allvars[varcount][0]:
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
            elif mymonthname == 'Dec' and int(myyear) != years_doubles_allvars[varcount][-1]:
                if 'winter' + ' ' + myyear + '-' + str(int(myyear)+1) not in seasons_list:
                    print('winter' + ' ' + myyear + '-' + str(int(myyear)+1))
                    seasons_list.append('winter' + ' ' + myyear + '-' + str(int(myyear)+1))
            elif mymonthname == 'Dec' and int(myyear) == years_doubles_allvars[varcount][-1]:
                if 'winter' + ' ' + myyear not in seasons_list:
                    print('winter' + ' ' + myyear)
                    seasons_list.append('winter' + ' ' + myyear)
        num_seasons = len(seasons_list)
        print('num_seasons = ' + str(num_seasons))
        print(seasons_list)
        seasonly_sums = np.zeros((np.prod(SOMsize), num_seasons))
        for iSOM in range(0, SOMsize[1]):  # row
            for jSOM in range(0, SOMsize[0]):  #col
                winning_node_indices = winning_nodes[jSOM, iSOM]
                print(str(jSOM) + ', ' + str(iSOM))
                print(len(winning_node_indices))
                #print(winning_node_indices)
                node_count = node_count + 1
                #if not 'node_numbers' in locals():  # if the dictionary for SOM node 2D indices and node count does not already exist, create it
                #    node_numbers = {(jSOM, iSOM): node_count}
                #else:  # if the the dictionary does exist, append new numbers to it
                #    node_numbers[(jSOM, iSOM)] = node_count
                #for imonth in range(0, num_months):
                year_month_str_count = -1
                for year_month_str in year_month_strs_allvars[varcount]:
                    sample_filename = 'MultiSensor_QPE_01H_Pass2_00.00_20210101-000000.nc'
                    #year_mon_str = myfile.split('/')[-1].split('.')[1][3:6+1] + '-' + myfile.split('/')[-1].split('.')[1][7:8+1]
                    #year_month_str = year_month_str[0:3+1] + year_month_str[5:6+1]  # get rid of the dash I inserted when originally making this variable, which I now realize was a mistake
                    #winning_node_indices_month = np.where((months_doubles[winning_node_indices] == imonth) & years_doubles[winning_node_indices] == )
                    #winning_node_indices_month = np.where(myfiles[np.array(winning_node_indices)].split('/')[-1].split('.')[1][3:8+1] == year_month_str)
                    #indices_month = np.where(myfiles.split('/')[-1].split('.')[1][3:8+1] == year_month_str)
                    indices_month = np.where(np.array(year_month_strs_all) == year_month_str)
                    #print('year_month_str = ' + year_month_str)
                    ##print('length of winning_node_indices_month = ' + str(len(winning_node_indices_month)))
                    #print(indices_month)
                    mymask = np.isin(indices_month, winning_node_indices)  # if an index in winning_node_indices is in indices_month, assign the corresponding element in a masked array of length len(indices_month) to True
                    winning_node_indices_month = np.array(indices_month)[mymask]  # only use the winning node indices for the current month
                    #print(winning_node_indices_month)
                    year_month_str_count = year_month_str_count + 1
                    monthly_sums[node_count, year_month_str_count] = monthly_sums[node_count, year_month_str_count] + len(winning_node_indices_month)
                season_str_count = -1
                for season_str in seasons_list:
                    indices_season = np.where(np.array(seasons_all) == season_str)
                    mymask2 = np.isin(indices_season, winning_node_indices)
                    winning_node_indices_season = np.array(indices_season)[mymask2]
                    season_str_count = season_str_count + 1
                    seasonly_sums[node_count, season_str_count] = seasonly_sums[node_count, season_str_count] + len(winning_node_indices_season)
        title_fontsize = 28
        axis_fontsize = 18
        annot_fontsize = 14
        ax = sns.heatmap(monthly_sums, annot=True, fmt='g', xticklabels=year_month_strs_allvars[varcount], yticklabels=np.arange(1, np.prod(SOMsize)+1), annot_kws={'size': annot_fontsize}, cbar_kws={'pad': 0.01})
        #ax.set_xlabel('month', fontsize=axis_fontsize)
        ax.set_ylabel('SOM node', fontsize=axis_fontsize)
        plt.yticks(rotation=0, fontsize=axis_fontsize)
        plt.xticks(rotation=45, ha='right', fontsize=axis_fontsize)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=axis_fontsize)
        ax.figure.subplots_adjust(left = 0.3)
#        plt.title('Maps per SOM node per month -- ' + varname_title + ' (99th percentile)', fontsize=title_fontsize)
#        plt.savefig('SOM_heatmap_' + myvarname + '_01H24Htrain_' + SOMsize_str + '_thresh_percentile_99th_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2.png', bbox_inches='tight', pad_inches=0.01)
        plt.title('Maps per SOM node per month -- ' + varname_title + ' (ari' + ari + ')', fontsize=title_fontsize)
        plt.savefig('SOM_heatmap_' + myvarname + '_01H24Htrain_' + SOMsize_str + '_thresh_ari' + ari + '_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2.png', bbox_inches='tight', pad_inches=0.01)

        fig = plt.figure(figsize=(27, 9))
        ax = sns.heatmap(seasonly_sums, annot=True, fmt='g', xticklabels=seasons_list, yticklabels=np.arange(1, np.prod(SOMsize)+1), annot_kws={'size': annot_fontsize}, cbar_kws={'pad': 0.01})
        ax.set_ylabel('SOM node', fontsize=axis_fontsize)
        plt.yticks(rotation=0, fontsize=axis_fontsize)
        plt.xticks(rotation=45, ha='right', fontsize=axis_fontsize)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=axis_fontsize)
        ax.figure.subplots_adjust(left = 0.3)
#        plt.title('Maps per SOM node per season -- ' + varname_title + ' (99th percentile)', fontsize=title_fontsize)
#        plt.savefig('SOM_heatmap_season_' + myvarname + '_01H24Htrain_' + SOMsize_str + '_thresh_percentile_99th_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2.png', bbox_inches='tight', pad_inches=0.01)
        plt.title('Maps per SOM node per season -- ' + varname_title + ' (ari' + ari + ')', fontsize=title_fontsize)
        plt.savefig('SOM_heatmap_season_' + myvarname + '_01H24Htrain_' + SOMsize_str + '_thresh_ari' + ari + '_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2.png', bbox_inches='tight', pad_inches=0.01)

        #win_positions = []
        #for myfile in myfiles:
            #win_position = varSOM.winner(myfile)  # get the winning SOM node for each output time
            #print(win_position)
            #win_positions.append(win_position)


        print(dataArrays[varcount].shape)
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
                winning_node_indices = winning_nodes[jSOM, iSOM]
                node_count = node_count + 1
#                dataArray_node = dataArray[winning_node_indices]
                dataArray_node = dataArrays[varcount][winning_node_indices]
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
                ax.title.set_text('Node ' + str(node_count + 1))
                # ax.outline_patch.set_alpha(0)  # plot outline/gridbox
                # ax.background_patch.set_alpha(0)  # plot background
                # plt.imshow(varSOM_plot, interpolation='none', aspect='auto')
        #cax = plt.axes([0.2, 0.525, 0.625, 0.025])
        cax = plt.axes([0.2, 0.075, 0.625, 0.025])
        cbar = fig.colorbar(ax_, cax=cax, orientation='horizontal', extend=my_var_cbarextend,
                            ticks=my_var_tick_levels, label=my_var_cbarlabel)#, pad=0.05)
        varname_title = myvarname.replace('_',' ')
        filename_first = myfiles[varcount][0]
        date_first = filename_first.split('/')[-1].split('.')[1][3:10+1]
        filename_last = myfiles[varcount][-1]
        date_last = filename_last.split('/')[-1].split('.')[1][3:10+1]
        #plt.suptitle(r'$\sigma$( ' + varname_title + ' ) -- ' + date_first + '-' + date_last, x=0.5125, y=0.9125, fontsize=14)
#        plt.suptitle(r'$\sigma$( ' + varname_title + ' ) (99th percentile) -- ' + date_first + '-' + date_last, x=0.5125, y=0.95, fontsize=14)
#        plt.savefig('SOM_' + myvarname + '_std_01H24Htrain_' + SOMsize_str + '_thresh_percentile_99th_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2.png')
        plt.suptitle(r'$\sigma$( ' + varname_title + ' ) (ari' + ari + ') -- ' + date_first + '-' + date_last, x=0.5125, y=0.95, fontsize=14)
        plt.savefig('SOM_' + myvarname + '_std_01H24Htrain_' + SOMsize_str + '_thresh_ari' + ari + '_' + date_first + '_' + date_last + '_' + myregion + '_test2_fixedtest2.png')


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
