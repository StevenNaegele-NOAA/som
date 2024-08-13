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


SOMsize_str = '5x5'
myregion = 'SC'
ari = '1yr'
precip_duration1 = '1hr'
precip_duration2 = '24hr'
# Needs to be read in outside of the function for some reason...?
#with open('SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count.pkl', 'rb') as f:
#    varnames, myfiles, myregion, percentile_90, percentile_95, percentile_99, percentile_99_5, percentile_99_9, percentile_99_99, indices_5mm_allvars, indices_10mm_allvars, indices_25mm_allvars, indices_50mm_allvars, indices_100mm_allvars, indices_150mm_allvars, indices_200mm_allvars, indices_250mm_allvars, indices_90th_allvars, indices_95th_allvars, indices_99th_allvars, indices_99_5th_allvars, indices_99_9th_allvars, indices_99_99th_allvars, datetimes_5mm_allvars, datetimes_10mm_allvars, datetimes_25mm_allvars, datetimes_50mm_allvars, datetimes_100mm_allvars, datetimes_150mm_allvars, datetimes_200mm_allvars, datetimes_250mm_allvars, datetimes_90th_allvars, datetimes_95th_allvars, datetimes_99th_allvars, datetimes_99_5th_allvars, datetimes_99_9th_allvars, datetimes_99_99th_allvars, count_5mm_allvars, count_10mm_allvars, count_25mm_allvars, count_50mm_allvars, count_100mm_allvars, count_150mm_allvars, count_200mm_allvars, count_250mm_allvars, count_90th_allvars, count_95th_allvars, count_99th_allvars, count_99_5th_allvars, count_99_9th_allvars, count_99_99th_allvars = pickle.load(f)
#with open('SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count_99th.pkl', 'rb') as f:
#    varnames, myfiles, myregion, percentile_99_allvars, indices_99th_allvars, datetimes_99th_allvars, count_99th_allvars = pickle.load(f)
with open('/scratch2/BMC/wrfruc/naegele/SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count_ari' + ari + '_2021.pkl', 'rb') as f:
    varnames, myfiles, myregion, indices_ari_allvars, datetimes_ari_allvars, count_ari_allvars = pickle.load(f)
print('len(indices_ari' + ari + precip_duration1 + '_allvars[0]) = ' + str(len(indices_ari_allvars[0])))
print('len(indices_ari' + ari + precip_duration2 + '_allvars[1]) = ' + str(len(indices_ari_allvars[1])))

#print('len(indices_99th_allvars[0]) = ' + str(len(indices_99th_allvars[0])))  # 25768
#print('len(indices_99th_allvars[1]) = ' + str(len(indices_99th_allvars[1])))  # 24864

#def mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapSize, myregion, d):
#def mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d):
def mysom2_multivar_thresh_newdomains(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d):
    print('Starting mysom function ...')
#    varSOMs = []
#    dataArrays_allSOMs = []
#    dataArray_train_allSOMs = []
#    for SOMsize in SOMsizes:
    print('Reading in data for SOM grid size ' + str(SOMsize))
    dataArrays = []
    months_doubles_allvars = []
    years_doubles_allvars = []
    year_month_strs_allvars = []
    year_month_counts_allvars = []

#    varnames_file = ['VAR_209_6_37_P0_L102_GLL0', 'VAR_209_6_41_P0_L102_GLL0']  # variable names in the netCDF file (VAR...37... = 01H QPE, VAR...41... = 24H QPE)
    varnames_file = ['precip', 'precip']  # variable names in the netCDF file (VAR...37... = 01H QPE, VAR...41... = 24H QPE)
#    varname_pickle = 'MultiSensor_QPE_01H_24H'  # variable name for the pickle file where output from this script will be saved
    varname_pickle = 'radaronly_QPE_01H_24H'  # variable name for the pickle file where output from this script will be saved

    datetimes_allvars = []

    maptedge = mapedges[0]
    mapbedge = mapedges[1]
    mapledge = mapedges[2]
    mapredge = mapedges[3]
    print('math.ceil((mapredge-mapledge)/10) = ' + str(math.ceil((mapredge-mapledge)/10)))
#    print('math.ceil((mapbedge-maptedge)/10) = ' + str(math.ceil((mapbedge-maptedge)/10)))
    print('math.ceil((maptedge-mapbedge)/10) = ' + str(math.ceil((maptedge-mapbedge)/10)))
#    print('np.prod(math.ceil((mapredge-mapledge)/10), math.ceil((mapbedge-maptedge)/10)) = ' + str(np.prod([math.ceil((mapredge-mapledge)/10), math.ceil((mapbedge-maptedge)/10)])))
    print('np.prod(math.ceil((mapredge-mapledge)/10), math.ceil((maptedge-mapbedge)/10)) = ' + str(np.prod([math.ceil((mapredge-mapledge)/10), math.ceil((maptedge-mapbedge)/10)])))

    for myvar in range(0, len(varnames)):  # STILL NEEDS TO BE MORE GENERALIZED
        print('Reading in data for variable ' + varnames[myvar])
        nMaps = len(myfiles[myvar])
        SOMsize_str = str(SOMsize[0]) + 'x' + str(SOMsize[1])  # SOM grid size expressed as a string (e.g. '4x3')
#        if myregion == 'NW':
#            maptedge = 1059#1  # top edge of the MRMS region
#            mapbedge = 530#1750  # bottom "                      "
#            mapledge = 1  # left "                      "
#            mapredge = 900#3500  # right "                      "
#        elif myregion == 'NC':
#            maptedge = 1059#1  # note that the top left corner is 1,1
#            mapbedge = 530#1750
#            mapledge = 451#1751
#            mapredge = 1350#5250
#        elif myregion == 'NE':
#            maptedge = 1059#1
#            mapbedge = 530#1750
#            mapledge = 900#3501
#            mapredge = 1799#7000
#        elif myregion == 'SW':
#            maptedge = 530#1751
#            mapbedge = 1#1059#3500
#            mapledge = 1
#            mapredge = 900#3500
#        elif myregion == 'SC':
#            maptedge = 530#1751
#            mapbedge = 1#1059#3500
#            mapledge = 451#1751
#            mapredge = 1350#5250
#        elif myregion == 'SE':
#            maptedge = 530#1751
#            mapbedge = 1#1059#3500
#            mapledge = 900#3501
#            mapredge = 1799#7000
#        mapedges = [maptedge, mapbedge, mapledge, mapredge]

        # Create an array to store the maps prior to SOM analysis
#            dataArray = np.empty((nMaps, np.prod(mapSize),))

        months = np.empty((nMaps, 1,))
        months_doubles = np.empty((nMaps, 1,))
        years = np.empty((nMaps, 1,))
        years_doubles = np.empty((nMaps, 1,))
        year_month = '000000'
        year_month_count = 0
        year_month_counts = np.empty((nMaps, 1,))
        year_month_strs = []

#            months_doubles_allvars = []
#            years_doubles_allvars = []
#            year_month_strs_allvars = []
#            year_month_counts_allvars = []

#        sample_filename = 'MultiSensor_QPE_01H_Pass2_00.00_20210101-000000.nc'
        sample_filename = '/scratch1/BMC/wrfruc/ejames/climo/mrms/202212/mrms_radaronly_2022123123.nc'
        fileCount = 0
        missing_count1 = 0 # missing file counter for var1
        missing_countmyvar = 0 # "                      " myvar
#            print(myvar)
            # For each file, retrieve the strings for the day, month, and year from the filename, and create a datetime object
#            for myfile in myfiles[0]:
#            for fileCount in range(0, len(myfiles[myvar])-missing_count1):
#            while fileCount < len(myfiles[myvar])-missing_count1:
        time_diff_full = date_last - date_first
        days, seconds = time_diff_full.days, time_diff_full.seconds
        hours_full = days*24 + seconds/3600
        datetimes_full = pandas.date_range(date_first,date_last,freq='h')
        datetimes_full = datetimes_full.format(formatter=lambda x: x.strftime('%Y%m%d-%H'))
#            print(datetimes_full)
#            datetimes_allvars = datetimes_full
#            missing_countfull = 0
#            while fileCount < hours_full-missing_countfull:
        datetimes_myvar = []
        for fileCount in range(0,len(myfiles[myvar])):

#                print('len(myfiles[0]) = ' + str(len(myfiles[0])))  # 716
#                print('len(myfiles[myvar]) = ' + str(len(myfiles[myvar])))  # 718
#                print('fileCount = ' + str(fileCount))  # 715, 713; 689
#                print('missing_count1 = ' + str(missing_count1))  # 4
#                print('missing_countmyvar = ' + str(missing_countmyvar))  # 2
#                print('missing_countfull = ' + str(missing_countfull))  # 6
#                myfile_var1 = myfiles[0][fileCount + missing_countmyvar]  # always files for the first variable

#                date_var1 = myfile_var1.split('/')[-1].split('.')[1][3:13+1]
#                date_var1 = datetime.datetime.strptime(date_var1, '%Y%m%d-%H')
#                print(date_var1)  # 2021-06-30 03:00:00, 2021-06-29 23:00:00

 #               myfile_myvar = myfiles[myvar][fileCount + missing_count1]  # files for any variable, including the first one (in which case no files will be missing)
            myfile_myvar = myfiles[myvar][fileCount]  # files for any variable, including the first one (in which case no files will be missing)
#            date_myvar = myfile_myvar.split('/')[-1].split('.')[1][3:13+1]
            date_myvar = myfile_myvar.split('/')[-1].split('_')[-1][0:9+1]
#            date_myvar = date_myvar[0:3+1] + date_myvar[4:5+1] + date_myvar[6:7+1] + '-' + date_myvar[9:10+1]
            date_myvar = date_myvar[0:3+1] + date_myvar[4:5+1] + date_myvar[6:7+1] + '-' + date_myvar[8:9+1]
#            print('date_myvar = ' + date_myvar + ', current time = ' + str(datetime.datetime.now()))
            datetimes_myvar.append(date_myvar)
#                date_myvar = datetime.datetime.strptime(date_myvar, '%Y%m%d-%H')
#                datetimes_myvar.append(date_myvar)
#                print('date_myvar = ' + str(date_myvar))  # 2021-06-30 03:00:00, 2021-06-29 23:00:00
#                print(type(date_myvar))
#            year = myfiles[myvar][fileCount].split('/')[-1].split('.')[1][3:6 + 1]
            year = myfiles[myvar][fileCount].split('/')[-1].split('_')[-1][0:3 + 1]
#            month = myfiles[myvar][fileCount].split('/')[-1].split('.')[1][7:8 + 1]
            month = myfiles[myvar][fileCount].split('/')[-1].split('_')[-1][4:5 + 1]
#                day = myfiles[myvar][fileCount].split('/')[-1].split('.')[1][9:10 + 1]
#                hour = myfiles[myvar][fileCount].split('/')[-1].split('.')[1][12:13 + 1]
#                ymd_h = year + month + day + '_' + hour + '0000'
            months[fileCount] = month
            months_doubles[fileCount] = int(month)
            years[fileCount] = year
            years_doubles[fileCount] = int(year)
            month_name = d[month]
            if year + '-' + month_name not in year_month_strs:
                year_month_strs.append(year + '-' + month_name)
            if year + month != year_month:
                year_month_count = year_month_count + 1
                year_month = year + month
            year_month_counts[fileCount] = year_month_count
        months_doubles_allvars.append(months_doubles)
        years_doubles_allvars.append(years_doubles)
        year_month_strs_allvars.append(year_month_strs)
        year_month_counts_allvars.append(year_month_counts)

#            print(type(datetimes_allvars))
#            print('Printing datetimes_myvar')
#            print(datetimes_myvar)
#            print(type(datetimes_myvar))
#            print(type(datetimes_myvar[0]))
        datetimes_allvars.append(datetimes_myvar)
#            print('Printing datetimes_allvars')
#            print(datetimes_allvars)
#            print('Printing all statement')
    for myvar in range(0, len(varnames)):  # STILL NEEDS TO BE MORE GENERALIZED
        dataArray = []        
        datetimes_common = []
        for dateCount in range(0, len(datetimes_full)):
#                print(datetimes_full[dateCount])
#                print(all(datetimes_full[dateCount] in datetimes_myvar for datetimes_myvar in datetimes_allvars))  # check if the current datetime from a list of all possible datetimes
            if all(datetimes_full[dateCount] in datetimes_myvar for datetimes_myvar in datetimes_allvars):  # check if the current datetime from a list of all possible datetimes
                                                                                                                 # is present for all variables
#                    print('datetimes_full = ' + str(datetimes_full[dateCount]))

#                if date_var1 != date_myvar:  # check if var1's datetime does not match myvar's datetime
#                    if date_var1 < date_myvar:  # if there is a mismatch and myvar is missing a file
#                        time_diff = date_myvar - date_var1
#                        days, seconds = time_diff.days, time_diff.seconds
#                        hours = days*24 + seconds/3600
#                        missing_countmyvar = missing_countmyvar + int(hours)  # add the difference b/n the datetimes (in hours) to the count
##                        missing_countfull = missing_countfull + int(hours)
#                    elif date_var1 > date_myvar:  # if there is a mismatch and var1 is missing a file
#                        time_diff = date_var1 - date_myvar
#                        days, seconds = time_diff.days, time_diff.seconds
#                        hours = days*24 + seconds/3600
#                        missing_count1 = missing_count1 + int(hours)  # "                                                          "
##                        missing_countfull = missing_countfull + int(hours)
#                year = myfiles[myvar][fileCount + missing_count1].split('/')[-1].split('.')[1][3:6 + 1]
#                month = myfiles[myvar][fileCount + missing_count1].split('/')[-1].split('.')[1][7:8 + 1]
#                day = myfiles[myvar][fileCount + missing_count1].split('/')[-1].split('.')[1][9:10 + 1]
#                hour = myfiles[myvar][fileCount + missing_count1].split('/')[-1].split('.')[1][12:13 + 1]
#                ymd_h = year + month + day + '_' + hour + '0000'
#                print(ymd_h)  # 20210630_030000, 20210629_210000

#                testvarread = xr.open_dataset(myfiles[myvar][fileCount + missing_count1], engine='netcdf4')#, chunks=2000)  # load the dataset into dask arrays, of size 2000 in each dimension
#                    fileCount_myvar = np.where(datetimes_allvars[myvar] == datetimes_full[dateCount])  # find out the index of the datetime for myvar that is the same as the current datetime from all datetimes
                fileCount_myvar = datetimes_allvars[myvar].index(datetimes_full[dateCount])  # find out the index of the datetime for myvar that is the same as the current datetime from all datetimes
#                print('fileCount_myvar = ' + str(fileCount_myvar) + ', current time = ' + str(datetime.datetime.now()))
                testvarread = xr.open_dataset(myfiles[myvar][fileCount_myvar], engine='netcdf4')#, chunks=2000)  # load the dataset into dask arrays, of size 2000 in each dimension
                # Read in variable data (note: variable name should be 'precip', but it is 'VAR_209...' when after converting original GRIB2 files to netCDF4
#                testvarMap = testvarread['VAR_209_6_37_P0_L102_GLL0'][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
#                testvarMap = testvarread['VAR_209_6_41_P0_L102_GLL0'][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
#                testvarMap = testvarread[varnames_file[myvar]][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
                testvarMap = testvarread[varnames_file[myvar]][np.arange(mapbedge, maptedge, 10), np.arange(mapledge, mapredge, 10)]
#                vectorOfMap = testvarMap.stack(point=["lon_0", "lat_0"])  # stacks each col end-to-end, making one long row (elim 1st dim)
                vectorOfMap = testvarMap.stack(point=["x", "y"])  # stacks each col end-to-end, making one long row (elim 1st dim)
#                vectorOfMap = np.ma.array(vectorOfMap)  # this dataset contains fill values, so in order to normalize and train on the data, the fill values need to be masked
#                for x in range(0, len(vectorOfMap)):
#                    if np.isnan(vectorOfMap[x]) or vectorOfMap[x] == -999 or vectorOfMap[x] == -9999:
#                        vectorOfMap[x] = np.ma.masked 
#                print('vectorOfMap at dateCount=' + str(dateCount) + ' = ' + str(vectorOfMap))

#                dataArray.append(vectorOfMap)
#                if max(vectorOfMap) >= percentile_99:
#                if np.max(vectorOfMap) >= percentile_99_allvars[myvar]:
                dataArray.append(vectorOfMap)
#                if datetimes_allvars[myvar][fileCount_myvar] not in datetimes_common:
                datetimes_common.append(datetimes_allvars[myvar][fileCount_myvar])

#                months[fileCount + missing_count1] = month
#                months_doubles[fileCount + missing_count1] = int(month)
#                years[fileCount + missing_count1] = year
#                years_doubles[fileCount + missing_count1] = int(year)
#                month_name = d[month]
#                if year + '-' + month_name not in year_month_strs:
#                    year_month_strs.append(year + '-' + month_name)
#                if year + month != year_month:
#                    year_month_count = year_month_count + 1
#                    year_month = year + month
#                year_month_counts[fileCount + missing_count1] = year_month_count
#                fileCount = fileCount + 1
        print('len(dataArray) = ' + str(len(dataArray)))  # 12533 --> 25768, 24864      716, 714; 696, 690
            #print('Printing dataArray')
            #print(dataArray)
        print('dataArray[-1] (pre-array) = ' + str(dataArray[-1]))  # [0. 0. 0. ... 0. -- --]
        dataArrays.append(np.ma.array(dataArray))
        print('dataArrays[myvar][-1] (post-array) = ' + str(dataArrays[myvar][-1]))  # [0. 0. 0. ... 0. -- --]
#            months_doubles_allvars.append(months_doubles)
#            years_doubles_allvars.append(years_doubles)
#            year_month_strs_allvars.append(year_month_strs)
#            year_month_counts_allvars.append(year_month_counts)
#    map_lats = testvarread['lat_0'][np.arange(maptedge, mapbedge, 10)]  # lat_0 is 3500 long
#    map_lats = testvarread['y'][np.arange(maptedge, mapbedge, 10)]  # lat_0 is 3500 long
    map_lats = testvarread['latitude'][np.arange(mapbedge, maptedge, 10), np.arange(mapledge, mapredge, 10)]  # lat_0 is 3500 long
#    map_lons = testvarread['lon_0'][np.arange(mapledge, mapredge, 10)]  # lon_0 is 7000 long
#    map_lons = testvarread['x'][np.arange(mapledge, mapredge, 10)]  # lon_0 is 7000 long
    map_lons = testvarread['longitude'][np.arange(mapbedge, maptedge, 10), np.arange(mapledge, mapredge, 10)]
    print('dataArrays[0][-1] = ' + str(dataArrays[0][-1]))  # [0. 0. 0. ... 0. nan nan]
    print('dataArrays[1][-1] = ' + str(dataArrays[1][-1]))  # [0. 0. 0. ... 0. nan nan]

#    # Find the common elements in two lists
#    def common_member(a, b):
#        a_set = set(a)
#        b_set = set(b)
#
#        if (a_set & b_set):
#            common_members = sorted(a_set & b_set)  # convert set to a list
#        else:
#            common_members = []
#
#        return common_members

#    indices_common = set.intersection(*[set(list_) for list_ in indices_99th_allvars])
#    print('len(indices_common) = ' + str(len(indices_common)))

    # For multiple variables, we would expect them to meet the defined threshold for different sets of times.
    # Thus, we want to ensure the dataArray trained on contains times when at least one variable meets the threshold (it could be multiple/all)
    # So, this for-loop saves only the indices associated with 1+ variables meeting the defined threshold
#    shared_indices = set(indices_99th_allvars[0])  # start the list of indices where the threshold is met for at least one variable, with the indices for the first variable (converted to a set)
#    shared_indices = set(indices_ari_allvars[0])  # start the list of indices where the threshold is met for at least one variable, with the indices for the first variable (converted to a set)
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
    print('len(shared_datetimes) = ' + str(len(shared_datetimes)))  # 25768; 24864
    print('shared_datetimes = ' + str(shared_datetimes))


#    # Save important variables to a file
#    print('Saving important variables to a file and returning to main script ...')
##    with open('SOM_' + SOMsize_str + '_thresh_percentile_99th_lr25_' + varname_pickle + '_' + myregion + '_vars_batch.pkl', 'wb') as f:  # Python 2: open(..., 'w')
#    with open('/scratch2/BMC/wrfruc/naegele/SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' + myregion + '_vars_batch_part1_2021.pkl', 'wb') as f:  # Python 2: open(..., 'w')
##[varSOMs, dataArray, map_lats, map_lons] = mysom2_multivar(myfiles, varnames, SOMsizes, mapSize, myregion, d)
##            pickle.dump([varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles,
##                         year_month_strs, year_month_counts, mapedges, myfiles, varname, SOMsize, mapSize, myregion], f)
##        pickle.dump([varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars,
##                     year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapSize, myregion], f)
##        pickle.dump([varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars,
##                     year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapedges, myregion], f)
#        pickle.dump([dataArrays, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars,
#                     year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapedges, myregion, shared_datetimes, datetimes_common], f)
#        # To open the file later
#        # with open('objs.pkl', 'rb') as f:  # Python 2: open(..., 'r') Python 2: open(..., 'rb')
#        #     obj0, obj1, obj2, ... = pickle.load(f)
#
##    return varSOM, dataArray, varSOM_nonzero, dataArray_nonzero, map_lats, map_lons, months_doubles, years_doubles, \
##    return varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles, \
##        year_month_strs, year_month_counts, mapedges
##    return varSOMs, dataArray_train_allSOMs, dataArrays_allSOMs, map_lats, map_lons
##    return varSOM, dataArray_train, dataArrays, percentile_99_allvars, indices_99th_allvars, count_99th_allvars, map_lats, map_lons
##    return varSOM, dataArray_train, dataArrays, map_lats, map_lons
#    return dataArrays, map_lats, map_lons




    varname_pickle = 'radaronly_QPE_01H_24H'  # variable name for the pickle file where output from this script will be saved
## Needs to be read in outside of the function for some reason...?
##with open('SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count.pkl', 'rb') as f:
##    varnames, myfiles, myregion, percentile_90, percentile_95, percentile_99, percentile_99_5, percentile_99_9, percentile_99_99, indices_5mm_allvars, indices_10mm_allvars, indices_25mm_allvars, indices_50mm_allvars, indices_100mm_allvars, indices_150mm_allvars, indices_200mm_allvars, indices_250mm_allvars, indices_90th_allvars, indices_95th_allvars, indices_99th_allvars, indices_99_5th_allvars, indices_99_9th_allvars, indices_99_99th_allvars, datetimes_5mm_allvars, datetimes_10mm_allvars, datetimes_25mm_allvars, datetimes_50mm_allvars, datetimes_100mm_allvars, datetimes_150mm_allvars, datetimes_200mm_allvars, datetimes_250mm_allvars, datetimes_90th_allvars, datetimes_95th_allvars, datetimes_99th_allvars, datetimes_99_5th_allvars, datetimes_99_9th_allvars, datetimes_99_99th_allvars, count_5mm_allvars, count_10mm_allvars, count_25mm_allvars, count_50mm_allvars, count_100mm_allvars, count_150mm_allvars, count_200mm_allvars, count_250mm_allvars, count_90th_allvars, count_95th_allvars, count_99th_allvars, count_99_5th_allvars, count_99_9th_allvars, count_99_99th_allvars = pickle.load(f)
##with open('SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count_99th.pkl', 'rb') as f:
##    varnames, myfiles, myregion, percentile_99_allvars, indices_99th_allvars, datetimes_99th_allvars, count_99th_allvars = pickle.load(f)
#with open('/scratch2/BMC/wrfruc/naegele/SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count_ari' + ari + '_2021.pkl', 'rb') as f:
#    varnames, myfiles, myregion, indices_ari_allvars, datetimes_ari_allvars, count_ari_allvars = pickle.load(f)
#print('len(indices_ari' + ari + precip_duration1 + '_allvars[0]) = ' + str(len(indices_ari_allvars[0])))#, flush=True)
#print('len(indices_ari' + ari + precip_duration2 + '_allvars[1]) = ' + str(len(indices_ari_allvars[1])))#, flush=True)

##print('len(indices_99th_allvars[0]) = ' + str(len(indices_99th_allvars[0])))  # 25768
##print('len(indices_99th_allvars[1]) = ' + str(len(indices_99th_allvars[1])))  # 24864

##def mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapSize, myregion, d):
##def mysom2_multivar_thresh(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d):
#def mysom2_multivar_thresh_part2(myfiles, date_first, date_last, varnames, SOMsize, mapedges, myregion, d, dataArrays, shared_datetimes, datetimes_common):
#    print('Starting mysom function (part 2) ...')#, flush=True)

#    maptedge, mapbedge, mapledge, mapredge = mapedges

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
        dataArray_normalized = [[x if not np.isnan(x) else -1 for x in y] for y in dataArray_normalized]  # replace any NaNs with a QPE that can never exist but is close enough to the data that it will not heavily influence training (e.g. -1)

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
#    varSOM = MiniSom(SOMsize[0], SOMsize[1], len(varnames)*np.prod([math.ceil((mapredge-mapledge)/10), math.ceil((maptedge-mapbedge)/10)]), sigma=1.75, learning_rate=0.125)  # optimal? [3x4, sigma=1.75, learning_rate=0.125, epochs=125]
    varSOM = MiniSom(SOMsize[0], SOMsize[1], len(varnames)*np.prod([math.ceil((mapredge-mapledge)/10), math.ceil((maptedge-mapbedge)/10)]), sigma=1.75, learning_rate=0.975)  # optimal? [3x4, sigma=1.75, learning_rate=0.125, epochs=125]
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
#    varSOM.train_batch(~np.isnan(dataArray_train), 500)  # ValueError: Received 9540 features, expected 9256. 2*90*53=9540 2*89*52=9256  ValueError: Received 19812 features, expected 9540.
    varSOM.train_batch(dataArray_train, 500)  # ValueError: Received 9540 features, expected 9256. 2*90*53=9540 2*89*52=9256  ValueError: Received 19812 features, expected 9540.
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
#    with open('/scratch2/BMC/wrfruc/naegele/SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' + myregion + '_vars_batch_newdomains_zeros_2021_small.pkl', 'wb') as f:  # Python 2: open(..., 'w')
    with open('/scratch2/BMC/wrfruc/naegele/SOM_' + SOMsize_str + '_thresh_ari' + ari + precip_duration1 + '_ari' + ari + precip_duration2 + '_' + varname_pickle + '_' + myregion + '_vars_batch_newdomains_zeros_2021.pkl', 'wb') as f:  # Python 2: open(..., 'w')
#[varSOMs, dataArray, map_lats, map_lons] = mysom2_multivar(myfiles, varnames, SOMsizes, mapSize, myregion, d)
#            pickle.dump([varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles,
#                         year_month_strs, year_month_counts, mapedges, myfiles, varname, SOMsize, mapSize, myregion], f)
#        pickle.dump([varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars,
#                     year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapSize, myregion], f)
#        pickle.dump([varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars,
#                     year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapedges, myregion], f)
        pickle.dump([varSOM, dataArrays_orig, dataArray_train, map_lats, map_lons, mapedges, myfiles, varnames, SOMsize, mapedges, myregion], f)
#        pickle.dump([varSOM, predictions, dataArrays_orig, dataArray_train, mapedges, myfiles, varnames, SOMsize, mapedges, myregion], f)
        # To open the file later
        # with open('objs.pkl', 'rb') as f:  # Python 2: open(..., 'r') Python 2: open(..., 'rb')
        #     obj0, obj1, obj2, ... = pickle.load(f)

#    return varSOM, dataArray, varSOM_nonzero, dataArray_nonzero, map_lats, map_lons, months_doubles, years_doubles, \
#    return varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles, \
#        year_month_strs, year_month_counts, mapedges
#    return varSOMs, dataArray_train_allSOMs, dataArrays_allSOMs, map_lats, map_lons
#    return varSOM, dataArray_train, dataArrays, percentile_99_allvars, indices_99th_allvars, count_99th_allvars, map_lats, map_lons
    return varSOM, dataArrays, map_lats, map_lons, dataArray_train
#    return varSOM, predictions, dataArray_train


