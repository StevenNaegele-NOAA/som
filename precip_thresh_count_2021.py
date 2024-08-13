# This script will create a SOM trained on MRMS MultiSensor 01H QPE

# Import necessary modules
import glob
import sys
import numpy as np
#from mysom2_multivar import mysom2_multivar  # my own function for training the SOM on multiple variables
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import xarray as xr
import datetime
import pickle
import pandas
import timeit

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
    varname = "_".join(filename_first.split("_", 2)[:2])  # split the filename string up to the second occurrence of "_", then combine the first two elements of the resulting list
    varnames.append(varname)
    varname_title = varname.replace('_',' ')
    varname_titles.append(varname_title)
#    date_first = filename_first.split('/')[-1].split('.')[1][3:10+1]
    date_first = filename_first.split('_')[-1].split('.')[0][0:7+1]
    date_firsts_strs.append(date_first)
    date_first = datetime.datetime.strptime(date_first, '%Y%m%d')
    date_firsts.append(date_first)
    filename_last = myfiles[myvar][-1]
    filename_lasts.append(filename_last)
#    date_last = filename_last.split('/')[-1].split('.')[1][3:10+1]
    date_last = filename_last.split('_')[-1].split('.')[0][0:7+1]
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
myregion = 'SC'#str(sys.argv[1])
print('My region =', myregion)

#SOMsize_dim1 = int(sys.argv[2])
#SOMsize_dim2 = int(sys.argv[3])
SOMsize = [5, 5]#[SOMsize_dim1, SOMsize_dim2]
print('SOMsize =', SOMsize)

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_nums = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
# d = dict(month_nums, month_names)
d = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
     '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
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


# Import necessary modules
#import numpy as np
#import xarray as xr
## from sklearn_som.som import SOM
#from minisom import MiniSom
#import pickle
#import timeit
#from sklearn.preprocessing import normalize
#import datetime
#import pandas

## Note: These files are of size 1799x1059 (HRRR dims), and the MRMS data provided by Eric is also at 1799x1059 (instead of native 7000x3500)
#ari10yr1hrread = xr.open_dataset('/scratch1/BMC/wrfruc/ejames/climo/ari/allusa_ari_10yr_1hr_3km.nc', engine='netcdf4')
##ari10yr1hrread = xr.open_dataset('/scratch1/BMC/wrfruc/ejames/climo/ari/allusa_ari_1yr_1hr_3km.nc', engine='netcdf4')
#print(ari10yr1hrread)
#ari10yr1hrMap = ari10yr1hrread['precip'][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
#print(ari10yr1hrMap)
#ari10yr1hr = ari10yr1hrMap.stack(point=["x", "y"])  # stacks each col end-to-end, making one long row (elim 1st dim)
#print(ari10yr1hr)
#print('NaN count in ari10yr1hr = ' + str(np.count_nonzero(np.isnan(ari10yr1hr))))
#ari10yr24hrread = xr.open_dataset('/scratch1/BMC/wrfruc/ejames/climo/ari/allusa_ari_10yr_24hr_3km.nc', engine='netcdf4')
#ari10yr24hrMap = ari10yr24hrread['precip']
#ari10yr24hr = ari10yr24hrMap.stack(point=["x", "y"])  # stacks each col end-to-end, making one long row (elim 1st dim)

#def mysom2_multivar(myfiles, date_first, date_last, varnames, SOMsize, mapSize, myregion, d):
#    print('Starting mysom function ...')
#    varSOMs = []
#    dataArrays_allSOMs = []
#    dataArray_train_allSOMs = []
#    for SOMsize in SOMsizes:
print('Reading in data for SOM grid size ' + str(SOMsize))
#dataArrays = []
months_doubles_allvars = []
years_doubles_allvars = []
year_month_strs_allvars = []
year_month_counts_allvars = []

varnames_file = ['VAR_209_6_37_P0_L102_GLL0', 'VAR_209_6_41_P0_L102_GLL0']  # variable names in the netCDF file (VAR...37... = 01H QPE, VAR...41... = 24H QPE)
varname_pickle = 'MultiSensor_QPE_01H_24H'  # variable name for the pickle file where output from this script will be saved

datetimes_allvars = []

dataArrays = []
indices_ari_allvars = []
datetimes_ari_allvars = []
count_ari_allvars = []

for myvar in range(0, len(varnames)):  # STILL NEEDS TO BE MORE GENERALIZED
    nMaps = len(myfiles[myvar])

    months = np.empty((nMaps, 1,))
    months_doubles = np.empty((nMaps, 1,))
    years = np.empty((nMaps, 1,))
    years_doubles = np.empty((nMaps, 1,))
    year_month = '000000'
    year_month_count = 0
    year_month_counts = np.empty((nMaps, 1,))
    year_month_strs = []

    #sample_filename = 'MultiSensor_QPE_01H_Pass2_00.00_20210101-000000.nc'
    sample_filename = '/scratch1/BMC/wrfruc/ejames/climo/mrms/202212/mrms_radaronly_2022123123.nc'
    fileCount = 0
    missing_count1 = 0 # missing file counter for var1
    missing_countmyvar = 0 # "                      " myvar
    time_diff_full = date_last - date_first
    days, seconds = time_diff_full.days, time_diff_full.seconds
    hours_full = days*24 + seconds/3600
    datetimes_full = pandas.date_range(date_first,date_last,freq='h')
    datetimes_full = datetimes_full.format(formatter=lambda x: x.strftime('%Y%m%d-%H'))
    datetimes_myvar = []
    for fileCount in range(0,len(myfiles[myvar])):
        myfile_myvar = myfiles[myvar][fileCount]  # files for any variable, including the first one (in which case no files will be missing)
#        date_myvar = myfile_myvar.split('/')[-1].split('.')[1][3:13+1]
        date_myvar = myfile_myvar.split('/')[-1].split('_')[-1][0:9+1]
#        date_myvar = date_myvar[0:3+1] + date_myvar[4:5+1] + date_myvar[6:7+1] + '-' + date_myvar[9:10+1]
        date_myvar = date_myvar[0:3+1] + date_myvar[4:5+1] + date_myvar[6:7+1] + '-' + date_myvar[8:9+1]
        datetimes_myvar.append(date_myvar)
#        year = myfiles[myvar][fileCount].split('/')[-1].split('.')[1][3:6 + 1]
#        month = myfiles[myvar][fileCount].split('/')[-1].split('.')[1][7:8 + 1]
#        months[fileCount] = month
#        months_doubles[fileCount] = int(month)
#        years[fileCount] = year
#        years_doubles[fileCount] = int(year)
#        month_name = d[month]
#        if year + '-' + month_name not in year_month_strs:
#            year_month_strs.append(year + '-' + month_name)
#        if year + month != year_month:
#            year_month_count = year_month_count + 1
#            year_month = year + month
#        year_month_counts[fileCount] = year_month_count
#    months_doubles_allvars.append(months_doubles)
#    years_doubles_allvars.append(years_doubles)
#    year_month_strs_allvars.append(year_month_strs)
#    year_month_counts_allvars.append(year_month_counts)
    datetimes_allvars.append(datetimes_myvar)


for myvar in range(0, len(varnames)):

    print('Reading in data for variable ' + varnames[myvar])
    nMaps = len(myfiles[myvar])
    SOMsize_str = str(SOMsize[0]) + 'x' + str(SOMsize[1])  # SOM grid size expressed as a string (e.g. '4x3')
    if myregion == 'NW':
        maptedge = 1  # top edge of the MRMS region
        mapbedge = 530#1750  # bottom "                      "
        mapledge = 1  # left "                      "
        mapredge = 900#3500  # right "                      "
    elif myregion == 'NC':
        maptedge = 1  # note that the top left corner is 1,1
        mapbedge = 530#1750
        mapledge = 451#1751
        mapredge = 1350#5250
    elif myregion == 'NE':
        maptedge = 1
        mapbedge = 530#1750
        mapledge = 900#3501
        mapredge = 1799#7000
    elif myregion == 'SW':
        maptedge = 530#1751
        mapbedge = 1059#3500
        mapledge = 1
        mapredge = 900#3500
    elif myregion == 'SC':
        maptedge = 530#1751
        mapbedge = 1059#3500
        mapledge = 451#1751
        mapredge = 1350#5250
    elif myregion == 'SE':
        maptedge = 530#1751
        mapbedge = 1059#3500
        mapledge = 900#3501
        mapredge = 1799#7000
    mapedges = [maptedge, mapbedge, mapledge, mapredge]

    ari_allvars = []
    precip_duration_allvars = []

    # Choose the average recurrence interval (1-year, 10-year, etc.) and precip durations (1-hour, 24-hour, etc.)
    ari = '1yr'
    precip_duration1 = '1hr'
    precip_duration2 = '24hr'
    # Note: These files are of size 1799x1059 (HRRR dims), and the MRMS data provided by Eric is also at 1799x1059 (instead of native 7000x3500)
    #ari10yr1hrread = xr.open_dataset('/scratch1/BMC/wrfruc/ejames/climo/ari/allusa_ari_10yr_1hr_3km.nc', engine='netcdf4')
    #ari10yr1hrread = xr.open_dataset('/scratch1/BMC/wrfruc/ejames/climo/ari/allusa_ari_1yr_1hr_3km.nc', engine='netcdf4')
    ariread1 = xr.open_dataset('/scratch1/BMC/wrfruc/ejames/climo/ari/allusa_ari_' + ari + '_' + precip_duration1 + '_3km.nc', engine='netcdf4')
#    print(ari10yr1hrread)
    #ari10yr1hrMap = ari10yr1hrread['precip'][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
    ariMap1 = ariread1['precip'][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
#    print(ari10yr1hrMap)
    #ari10yr1hr = ari10yr1hrMap.stack(point=["x", "y"])  # stacks each col end-to-end, making one long row (elim 1st dim)
    ari1 = ariMap1.stack(point=["x", "y"])  # stacks each col end-to-end, making one long row (elim 1st dim)
    #print(ari10yr1hr)
    #print('NaN count in ari10yr1hr = ' + str(np.count_nonzero(np.isnan(ari10yr1hr))))
    print(ari1)
    print('NaN count in ari' + ari + precip_duration1 + ' = ' + str(np.count_nonzero(np.isnan(ari1))))
    ari_allvars.append(ari1)
    precip_duration_allvars.append(precip_duration1)
    #ari10yr24hrread = xr.open_dataset('/scratch1/BMC/wrfruc/ejames/climo/ari/allusa_ari_10yr_24hr_3km.nc', engine='netcdf4')
    ariread2 = xr.open_dataset('/scratch1/BMC/wrfruc/ejames/climo/ari/allusa_ari_' + ari + '_' + precip_duration2 + '_3km.nc', engine='netcdf4')
    #ari10yr24hrMap = ari10yr24hrread['precip']
    ariMap2 = ariread2['precip'][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
    #ari10yr24hr = ari10yr24hrMap.stack(point=["x", "y"])  # stacks each col end-to-end, making one long row (elim 1st dim)
    ari2 = ariMap2.stack(point=["x", "y"])  # stacks each col end-to-end, making one long row (elim 1st dim)
    print(ari2)
    print('NaN count in ari' + ari + precip_duration2 + ' = ' + str(np.count_nonzero(np.isnan(ari2))))
    ari_allvars.append(ari2)
    precip_duration_allvars.append(precip_duration2)
    

#    with open('SOM_' + SOMsize_str + '_learningrate25_MultiSensor_QPE_01H_24H_' + myregion + '_vars_batch.pkl', 'rb') as f:
##    varSOM, dataArray, varSOM_nonzero, dataArray_nonzero, map_lats, map_lons, months_doubles, years_doubles, \
##    varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles, \
#        varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars, \
#        year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapSize, myregion = pickle.load(f)
    dataArray = []
    indices_ari = []
    datetimes_ari = []
    count_ari = 0
#    ari_closest_dist = 9999.
#    ari_closest_ind = -9999
#    testvarread = xr.open_mfdataset(myfiles[myvar][fileCount_myvar], engine='netcdf4', chunks=100)  # load the dataset into dask arrays, of size 2000 in each dimension
    for dateCount in range(0, len(datetimes_full)):
#        print(datetimes_full)
#        print(datetimes_allvars)
        if all(datetimes_full[dateCount] in datetimes_myvar for datetimes_myvar in datetimes_allvars):  # check if the current datetime from a list of all possible datetimes
                                                                                                        # is present for all variables
            print(datetimes_full[dateCount])
            fileCount_myvar = datetimes_allvars[myvar].index(datetimes_full[dateCount])  # find out the index of the datetime for myvar that is the same as the current datetime from all datetimes
            print('fileCount_myvar = ' + str(fileCount_myvar) + ', current time = ' + str(datetime.datetime.now()))
            testvarread = xr.open_dataset(myfiles[myvar][fileCount_myvar], engine='netcdf4')#, chunks={"latitude":100})#, "longitude":50})#, chunks=100)  # load the dataset into dask arrays, of size 2000 in each dimension
            #print('testvarread = ' + str(testvarread))
#            lonMap = testvarread['lon_0'][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
#            vectorOflonMap = lonMap.stack(point=["lon_0", "lat_0"])  # stacks each col end-to-end, making one long row (elim 1st dim)
#            latMap = testvarread['lat_0'][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
#            vectorOflatMap = latMap.stack(point=["lon_0", "lat_0"])  # stacks each col end-to-end, making one long row (elim 1st dim)
#            testvarMap = testvarread[varnames_file[myvar]][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
            testvarMap = testvarread['precip'][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
#            print('testvarMap[0:52][0:89] = ' + str(testvarMap[0:52][0:89]))
#            vectorOfMap = testvarMap.stack(point=["lon_0", "lat_0"])  # stacks each col end-to-end, making one long row (elim 1st dim)
            vectorOfMap = testvarMap.stack(point=["x", "y"])  # stacks each col end-to-end, making one long row (elim 1st dim)
#            print('vectorOfMap[2300:2400] = ' + str(vectorOfMap[2300:2400]))
#            print('max(vectorOfMap) = ' + str(np.max(vectorOfMap)))
#            print('ari10yr1hr[2300:2400] = ' + str(ari10yr1hr[2300:2400]))
#            print('max(ari10yr1hr) = ' + str(np.max(ari10yr1hr)))
#            print('len(vectorOfMap) = ' + str(len(vectorOfMap)))  # 4770
#            print('len(ari10yr1hr) = ' + str(len(ari10yr1hr)))  # 4770
#            starttime = timeit.default_timer()
            for gridpoint in range(0, len(vectorOfMap)):
                ari_gridpoint = ari_allvars[myvar][gridpoint]
#                print('ari10yr1hr_gridpoint = ' + str(ari10yr1hr_gridpoint))  # all NaNs?
#                print('vectorOfMap[gridpoint] = ' + str(vectorOfMap[gridpoint]))
                if vectorOfMap[gridpoint] > ari_gridpoint:
                    print('vectorOfMap[gridpoint] > ari' + ari + precip_duration_allvars[myvar] + '[gridpoint]')#= ' + str(vectorOfMap[gridpoint]))
                    dataArray.append(vectorOfMap)
                    indices_ari.append(dateCount)
                    datetimes_ari.append(datetimes_full[dateCount])
                    count_ari = count_ari + 1
                    break
            else:
                # will be called if the previous loop did not end with a `break`
                continue
#            print("The time spent iterating through the grid points is :", timeit.default_timer() - starttime)
    print('len(dataArray) = ' + str(len(dataArray)))  # 12533 --> 25768, 24864      716, 714; 696, 690
    print('len(indices_ari' + ari + precip_duration_allvars[myvar] + ') = ' + str(len(indices_ari)))  # 
    print('len(datetimes_ari' + ari + precip_duration_allvars[myvar] + ') = ' + str(len(datetimes_ari)))  # 
    print('len(count_ari' + ari + precip_duration_allvars[myvar] + ') = ' + str(count_ari))  # 
    dataArrays.append(np.array(dataArray))
    indices_ari_allvars.append(indices_ari)
    datetimes_ari_allvars.append(datetimes_ari)
    count_ari_allvars.append(count_ari)

#percentile_99_allvars = []
#indices_5mm_allvars = []  # ~0.2 in/hr
#indices_10mm_allvars = []  # ~0.4 in/hr
#indices_25mm_allvars = []  # ~1 in/hr
#indices_50mm_allvars = []  # ~2 in/hr
#indices_100mm_allvars = []  # ~4 in/hr
#indices_150mm_allvars = []  # ~6 in/hr
#indices_200mm_allvars = []  # ~8 in/hr
#indices_250mm_allvars = []  # ~10 in/hr
#indices_90th_allvars = []  # 90th percentile
#indices_95th_allvars = []  # 95th percentile
#indices_99th_allvars = []  # 99th percentile
#indices_99_5th_allvars = []  # 99.5th percentile
#indices_99_9th_allvars = []  # 99.9th percentile
#indices_99_99th_allvars = []  # 99.99th percentile
#datetimes_5mm_allvars = []
#datetimes_10mm_allvars = []
#datetimes_25mm_allvars = []
#datetimes_50mm_allvars = []
#datetimes_100mm_allvars = []
#datetimes_150mm_allvars = []
#datetimes_200mm_allvars = []
#datetimes_250mm_allvars = []
#datetimes_90th_allvars = []
#datetimes_95th_allvars = []
#datetimes_99th_allvars = []
#datetimes_99_5th_allvars = []
#datetimes_99_9th_allvars = []
#datetimes_99_99th_allvars = []
#count_5mm_allvars = []
#count_10mm_allvars = []
#count_25mm_allvars = []
#count_50mm_allvars = []
#count_100mm_allvars = []
#count_150mm_allvars = []
#count_200mm_allvars = []
#count_250mm_allvars = []
#count_90th_allvars = []
#count_95th_allvars = []
#count_99th_allvars = []
#count_99_5th_allvars = []
#count_99_9th_allvars = []
#count_99_99th_allvars = []
#for myvar in range(0, len(varnames)):  # STILL NEEDS TO BE MORE GENERALIZED

#    print('len(dataArrays[myvar]) = ' + str(len(dataArrays[myvar])))
#    print('len(dataArrays[myvar][0]) = ' + str(len(dataArrays[myvar][0])))
#    print('max(dataArrays_orig[myvar]) = ' + str(max(np.array(dataArrays_orig[myvar]))))

    # potential threshold for heavy precip (as defined by EPA: https://www.epa.gov/sites/default/files/2021-04/documents/heavy-precip_td.pdf)
    # note: to avoid the influence of too many 0's, only calculate the 90th percentile on the domain-wide precip maxima for each time
#    percentile_90 = np.percentile(dataArrays_orig[myvar], 90)                          # 0.0 mm,                       26182 times;    5.800000190734863 mm,   26160 times
#    percentile_95 = np.percentile(dataArrays_orig[myvar], 95)                          #
#    percentile_99 = np.percentile(dataArrays_orig[myvar], 99)                           # 2.200000047683716 mm,         25768 times;    38.900001525878906 mm,  24864 times
#    percentile_99_5 = np.percentile(dataArrays_orig[myvar], 99.5)                      #
#    percentile_99_9 = np.percentile(dataArrays_orig[myvar], 99.9)                      # 12.40000057220459 mm,         22082 times;    91.9000015258789 mm,    20362 times
#    percentile_99_99 = np.percentile(dataArrays_orig[myvar], 99.99)                    # 31.399999618530273 mm,        15500 times;    175.6999969482422 mm,   7595 times
    #percentile_90 = np.percentile(np.max(dataArrays_orig[myvar],axis=1), 90)           # 68.0 mm,                      2626 times;     252.0 mm,               2622 times
    #percentile_99 = np.percentile(np.max(dataArrays_orig[myvar],axis=1), 99)           # 100.5999984741211 mm,         263 times;      464.3380148315427 mm,   262 times
    #percentile_99_9 = np.percentile(np.max(dataArrays_orig[myvar],axis=1), 99.9)       # 142.54750000000786 mm,        27 times;       799.2552266235376 mm,   27 times
    #percentile_99_99 = np.percentile(np.max(dataArrays_orig[myvar],axis=1), 99.99)     # 196.4220816100968 mm,         3 times;        877.5874397521912 mm,   3 times
#    percentile_99_allvars.append(percentile_99)
    #dataArray_domain_sums = np.zeros(len(dataArrays_orig[myvar]))
    #dataArray_domain_maxs = np.zeros(len(dataArrays_orig[myvar]))
    #for dateCount in range(0, len(dataArrays_orig[myvar])):
    #    #dataArray_domain_sums[dateCount] = np.sum(dataArrays_orig[myvar][dateCount])
    #    dataArray_domain_maxs[dateCount] = np.max(dataArrays_orig[myvar][dateCount])
    ##nonzero_indices = np.where(dataArray_domain_sums > 1.0 )  # get times when there is measurable precip somewhere in the region
    #nonzero_indices = np.where(dataArray_domain_maxs > 1.0 )  # get times when there is measurable precip somewhere in the region
    #print('len(dataArrays_orig[myvar]) = ' + str(len(dataArrays_orig[myvar])))  # 26182
    #print('len(dataArrays_orig[myvar][0]) = ' + str(len(dataArrays_orig[myvar][0])))  # 61250
    #print('np.sum(dataArrays_orig[myvar][0]) = ' + str(np.sum(dataArrays_orig[myvar][0])))  # 17723.9
    #print('indices of times with precip > 1.0 mm = ' + str(nonzero_indices))  #
    #print('first index of times with precip > 1.0 mm = ' + str(nonzero_indices[0]))  #
    #print('number of times with precip > 1.0 mm = ' + str(len(nonzero_indices[0])))  # 26143 (sums)
    #percentile_90 = np.percentile(dataArrays_orig[myvar][nonzero_indices], 90)         # 0.800000011920929 mm,         26089 times;    20.700000762939453 mm,  25768 times
    #percentile_99 = np.percentile(dataArrays_orig[myvar][nonzero_indices], 99)         # 5.0 mm,                       24712 times;    69.0999984741211 mm,    22781 times
    #percentile_99_9 = np.percentile(dataArrays_orig[myvar][nonzero_indices], 99.9)     # 10.975100142479327 mm,        22601 times;    93.70000457763672 mm,   20110 times
    #percentile_99_99 = np.percentile(dataArrays_orig[myvar][nonzero_indices], 99.99)   # 29.362770667641495 mm,        16246 times;    117.625142861172 mm,    16354 times
    #gridpoint_percentile_90 = np.zeros(len(dataArrays_orig[myvar][0]))
    #gridpoint_percentile_99 = np.zeros(len(dataArrays_orig[myvar][0]))
    #gridpoint_percentile_99_9 = np.zeros(len(dataArrays_orig[myvar][0]))
    #gridpoint_percentile_99_99 = np.zeros(len(dataArrays_orig[myvar][0]))
    #for gridpoint in range(0, len(dataArrays_orig[myvar][0])):
    #    gridpoint_percentile_90[gridpoint] = np.percentile(np.array(dataArrays_orig)[myvar,:,gridpoint], 90)  # 90th percentile for each grid point across time
    #    gridpoint_percentile_99[gridpoint] = np.percentile(np.array(dataArrays_orig)[myvar,:,gridpoint], 99)  # 99th percentile for each grid point across time
    #    gridpoint_percentile_99_9[gridpoint] = np.percentile(np.array(dataArrays_orig)[myvar,:,gridpoint], 99.9)  # 99.9th percentile for each grid point across time
    #    gridpoint_percentile_99_99[gridpoint] = np.percentile(np.array(dataArrays_orig)[myvar,:,gridpoint], 99.99)  # 99.99th percentile for each grid point across time

#    dataArray = []
#    indices_5mm = []
#    indices_10mm = []
#    indices_25mm = []
#    indices_50mm = []
#    indices_100mm = []
#    indices_150mm = []
#    indices_200mm = []
#    indices_250mm = []
#    indices_90th = []
#    indices_95th = []
#    indices_99th = []
#    indices_99_5th = []
#    indices_99_9th = []
#    indices_99_99th = []
#    datetimes_5mm = []
#    datetimes_10mm = []
#    datetimes_25mm = []
#    datetimes_50mm = []
#    datetimes_100mm = []
#    datetimes_150mm = []
#    datetimes_200mm = []
#    datetimes_250mm = []
#    datetimes_90th = []
#    datetimes_95th = []
#    datetimes_99th = []
#    datetimes_99_5th = []
#    datetimes_99_9th = []
#    datetimes_99_99th = []
#    count_5mm = 0
#    count_10mm = 0
#    count_25mm = 0
#    count_50mm = 0
#    count_100mm = 0
#    count_150mm = 0
#    count_200mm = 0
#    count_250mm = 0
#    count_90th = 0
#    count_95th = 0
#    count_99th = 0
#    count_99_5th = 0
#    count_99_9th = 0
#    count_99_99th = 0
#    for dateCount in range(0, len(datetimes_full)):
#    for dateCount in range(0, len(dataArrays[myvar])):
#        if all(datetimes_full[dateCount] in datetimes_myvar for datetimes_myvar in datetimes_allvars):  # check if the current datetime from a list of all possible datetimes
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

#            fileCount_myvar = datetimes_allvars[myvar].index(datetimes_full[dateCount])  # find out the index of the datetime for myvar that is the same as the current datetime from all datetimes
#            print('fileCount_myvar = ' + str(fileCount_myvar))
#            testvarread = xr.open_dataset(myfiles[myvar][fileCount_myvar], engine='netcdf4')#, chunks=2000)  # load the dataset into dask arrays, of size 2000 in each dimension
#                # Read in variable data (note: variable name should be 'precip', but it is 'VAR_209...' when after converting original GRIB2 files to netCDF4
##                testvarMap = testvarread['VAR_209_6_37_P0_L102_GLL0'][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
##                testvarMap = testvarread['VAR_209_6_41_P0_L102_GLL0'][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
##                testvarMap = testvarread[varnames_file[myvar]][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
#            testvarMap = testvarread[varnames_file[myvar]][np.arange(maptedge, mapbedge, 10), np.arange(mapledge, mapredge, 10)]
#            vectorOfMap = testvarMap.stack(point=["lon_0", "lat_0"])  # stacks each col end-to-end, making one long row (elim 1st dim)

#        vectorOfMap = dataArrays[myvar][dateCount]

#        if max(vectorOfMap) >= 5.0:  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_5mm.append(dateCount)
#            datetimes_5mm.append(datetimes_full[dateCount])
#            count_5mm = count_5mm + 1
#        if max(vectorOfMap) >= 10.0:  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_10mm.append(dateCount)
#            datetimes_10mm.append(datetimes_full[dateCount])
#            count_10mm = count_10mm + 1
#        if max(vectorOfMap) >= 25.0:  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_25mm.append(dateCount)
#            datetimes_25mm.append(datetimes_full[dateCount])
#            count_25mm = count_25mm + 1
#        if max(vectorOfMap) >= 50.0:  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_50mm.append(dateCount)
#            datetimes_50mm.append(datetimes_full[dateCount])
#            count_50mm = count_50mm + 1
#        if max(vectorOfMap) >= 100.0:  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_100mm.append(dateCount)
#            datetimes_100mm.append(datetimes_full[dateCount])
#            count_100mm = count_100mm + 1
#        if max(vectorOfMap) >= 150.0:  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_150mm.append(dateCount)
#            datetimes_150mm.append(datetimes_full[dateCount])
#            count_150mm = count_150mm + 1
#        if max(vectorOfMap) >= 200.0:  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_200mm.append(dateCount)
#            datetimes_200mm.append(datetimes_full[dateCount])
#            count_200mm = count_200mm + 1
#        if max(vectorOfMap) >= 250.0:  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_250mm.append(dateCount)
#            datetimes_250mm.append(datetimes_full[dateCount])
#            count_250mm = count_250mm + 1
#        if max(vectorOfMap) >= percentile_90:  # mm / hr (if var1) or / 24 hr (if var2)
#        #if any(max(vectorOfMap) >= gridpoint_percentile_90):  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_90th.append(dateCount)
#            datetimes_90th.append(datetimes_full[dateCount])
#            count_90th = count_90th + 1
#        if max(vectorOfMap) >= percentile_95:  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_95th.append(dateCount)
#            datetimes_95th.append(datetimes_full[dateCount])
#            count_95th = count_95th + 1
#        if max(vectorOfMap) >= percentile_99:  # mm / hr (if var1) or / 24 hr (if var2)
#        #if any(max(vectorOfMap) >= gridpoint_percentile_99):  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_99th.append(dateCount)
#            datetimes_99th.append(datetimes_full[dateCount])
#            count_99th = count_99th + 1
#        if max(vectorOfMap) >= percentile_99_5:  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_99_5th.append(dateCount)
#            datetimes_99_5th.append(datetimes_full[dateCount])
#            count_99_5th = count_99_5th + 1
#        if max(vectorOfMap) >= percentile_99_9:  # mm / hr (if var1) or / 24 hr (if var2)
#        #if any(max(vectorOfMap) >= gridpoint_percentile_99_9):  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_99_9th.append(dateCount)
#            datetimes_99_9th.append(datetimes_full[dateCount])
#            count_99_9th = count_99_9th + 1
#        if max(vectorOfMap) >= percentile_99_99:  # mm / hr (if var1) or / 24 hr (if var2)
#        #if any(max(vectorOfMap) >= gridpoint_percentile_99_99):  # mm / hr (if var1) or / 24 hr (if var2)
#            indices_99_99th.append(dateCount)
#            datetimes_99_99th.append(datetimes_full[dateCount])
#            count_99_99th = count_99_99th + 1

#            dataArray.append(vectorOfMap)

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
#    print('len(dataArray) = ' + str(len(dataArray)))  # 716, 714; 696, 690
#    dataArrays.append(np.array(dataArray))

#    indices_5mm_allvars.append(indices_5mm)
#    indices_10mm_allvars.append(indices_10mm)
#    indices_25mm_allvars.append(indices_25mm)
#    indices_50mm_allvars.append(indices_50mm)
#    indices_100mm_allvars.append(indices_100mm)
#    indices_150mm_allvars.append(indices_150mm)
#    indices_200mm_allvars.append(indices_200mm)
#    indices_250mm_allvars.append(indices_250mm)
#    indices_90th_allvars.append(indices_90th)
#    indices_95th_allvars.append(indices_95th)
#    indices_99th_allvars.append(indices_99th)
#    indices_99_5th_allvars.append(indices_99_5th)
#    indices_99_9th_allvars.append(indices_99_9th)
#    indices_99_99th_allvars.append(indices_99_99th)
#    datetimes_5mm_allvars.append(datetimes_5mm)
#    datetimes_10mm_allvars.append(datetimes_10mm)
#    datetimes_25mm_allvars.append(datetimes_25mm)
#    datetimes_50mm_allvars.append(datetimes_50mm)
#    datetimes_100mm_allvars.append(datetimes_100mm)
#    datetimes_150mm_allvars.append(datetimes_150mm)
#    datetimes_200mm_allvars.append(datetimes_200mm)
#    datetimes_250mm_allvars.append(datetimes_250mm)
#    datetimes_90th_allvars.append(datetimes_90th)
#    datetimes_95th_allvars.append(datetimes_95th)
#    datetimes_99th_allvars.append(datetimes_99th)
#    datetimes_99_5th_allvars.append(datetimes_99_5th)
#    datetimes_99_9th_allvars.append(datetimes_99_9th)
#    datetimes_99_99th_allvars.append(datetimes_99_99th)
#    count_5mm_allvars.append(count_5mm)
#    count_10mm_allvars.append(count_10mm)
#    count_25mm_allvars.append(count_25mm)
#    count_50mm_allvars.append(count_50mm)
#    count_100mm_allvars.append(count_100mm)
#    count_150mm_allvars.append(count_150mm)
#    count_200mm_allvars.append(count_200mm)
#    count_250mm_allvars.append(count_250mm)
#    count_90th_allvars.append(count_90th)
#    count_95th_allvars.append(count_95th)
#    count_99th_allvars.append(count_99th)
#    count_99_5th_allvars.append(count_99_5th)
#    count_99_9th_allvars.append(count_99_9th)
#    count_99_99th_allvars.append(count_99_99th)
#    print('Domain wide ' + varnames[myvar] + ' is >= 5 mm ' + str(count_5mm) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
#    print('Domain wide ' + varnames[myvar] + ' is >= 10 mm ' + str(count_10mm) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
#    print('Domain wide ' + varnames[myvar] + ' is >= 25 mm ' + str(count_25mm) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
#    print('Domain wide ' + varnames[myvar] + ' is >= 50 mm ' + str(count_50mm) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
#    print('Domain wide ' + varnames[myvar] + ' is >= 100 mm ' + str(count_100mm) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
#    print('Domain wide ' + varnames[myvar] + ' is >= 150 mm ' + str(count_150mm) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
#    print('Domain wide ' + varnames[myvar] + ' is >= 200 mm ' + str(count_200mm) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
#    print('Domain wide ' + varnames[myvar] + ' is >= 250 mm ' + str(count_250mm) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
#    print('Domain wide ' + varnames[myvar] + ' is >= the 90th percentile (' + str(percentile_90) + ' mm) ' + str(count_90th) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
#    print('Domain wide ' + varnames[myvar] + ' is >= the 95th percentile (' + str(percentile_95) + ' mm) ' + str(count_95th) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
#    print('Domain wide ' + varnames[myvar] + ' is >= the 99th percentile (' + str(percentile_99) + ' mm) ' + str(count_99th) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
#    print('Domain wide ' + varnames[myvar] + ' is >= the 99.5th percentile (' + str(percentile_99_5) + ' mm) ' + str(count_99_5th) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
#    print('Domain wide ' + varnames[myvar] + ' is >= the 99.9th percentile (' + str(percentile_99_9) + ' mm) ' + str(count_99_9th) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
#    print('Domain wide ' + varnames[myvar] + ' is >= the 99.99th percentile (' + str(percentile_99_99) + ' mm) ' + str(count_99_99th) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
    #print('Domain wide ' + varnames[myvar] + ' is >= the gridpoint 90th percentile (' + str(np.mean(gridpoint_percentile_90)) + ' mm on average) ' + str(count_90th) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
    #print('Domain wide ' + varnames[myvar] + ' is >= the gridpoint 99th percentile (' + str(np.mean(gridpoint_percentile_99)) + ' mm on average) ' + str(count_99th) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
    #print('Domain wide ' + varnames[myvar] + ' is >= the gridpoint 99.9th percentile (' + str(np.mean(gridpoint_percentile_99_9)) + ' mm on average) ' + str(count_99_9th) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
    #print('Domain wide ' + varnames[myvar] + ' is >= the gridpoint 99.99th percentile (' + str(np.mean(gridpoint_percentile_99_99)) + ' mm on average) ' + str(count_99_99th) + ' times out of ' + str(len(myfiles[myvar])) + ' total files')
##            months_doubles_allvars.append(months_doubles)
##            years_doubles_allvars.append(years_doubles)
##            year_month_strs_allvars.append(year_month_strs)
##            year_month_counts_allvars.append(year_month_counts)
#map_lats = testvarread['lat_0'][np.arange(maptedge, mapbedge, 10)]  # lat_0 is 3500 long
#map_lons = testvarread['lon_0'][np.arange(mapledge, mapredge, 10)]  # lon_0 is 7000 long
#dataArrays_orig = dataArrays
#for dataArray in dataArrays:
#    dataArray = normalize(dataArray, norm='l1', axis=1)  # L1 normalization each row independently, and normalize each sample (across time?) instead of each feature (across space?)
#print(dataArrays)
#print('NaN count for dataArrays = ' + str(np.count_nonzero(np.isnan(dataArrays))))
#dataArray_train = np.hstack(dataArrays)  # dataArrays needs to be an array for some reason? (even though in testing a list was fine)
#print(dataArray_train)
#print('NaN count for dataArray_train = ' + str(np.count_nonzero(np.isnan(dataArray_train))))
#
#print('Configuring and training SOM ...')
##        varSOM = MiniSom(SOMsize[0], SOMsize[1], np.prod(mapSize), sigma=0.3, learning_rate=0.5)
##    varSOM = MiniSom(SOMsize[0], SOMsize[1], len(varnames)*np.prod(mapSize), sigma=1.75, learning_rate=0.125)  # optimal? [3x4, sigma=1.75, learning_rate=0.125, epochs=125]
#varSOM = MiniSom(SOMsize[0], SOMsize[1], len(varnames)*np.prod(mapSize), sigma=1.75, learning_rate=0.25)  # increasing learning rate from "optimal" value in an attempt to figure out why
#                                                                                                              # some nodes in larger grids have a few cases for the first month and then none afterward
##        varSOM = MiniSom(SOMsize[0], SOMsize[1], np.prod(mapSize), sigma=1.0, learning_rate=0.125)  # alt sigma [spread of neighborhood function] (which is actually the default)
##        varSOM.train_batch(dataArray_train, 1000)
#varSOM.train_batch(dataArray_train, 500)
##        varSOMs.append(varSOM)
#
    # Save important variables to a file
print('Saving important variables to a file ...')
#with open('SOM_' + SOMsize_str + '_learningrate25_' + varname_pickle + '_' + myregion + '_vars_batch.pkl', 'wb') as f:  # Python 2: open(..., 'w')
#with open('SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count_99th.pkl', 'wb') as f:  # Python 2: open(..., 'w')
with open('/scratch2/BMC/wrfruc/naegele/SOM_' + SOMsize_str + '_multivar_' + myregion + '_precip_thresh_count_ari' + ari + '_2021.pkl', 'wb') as f:  # Python 2: open(..., 'w')
##[varSOMs, dataArray, map_lats, map_lons] = mysom2_multivar(myfiles, varnames, SOMsizes, mapSize, myregion, d)
##            pickle.dump([varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles,
##                         year_month_strs, year_month_counts, mapedges, myfiles, varname, SOMsize, mapSize, myregion], f)
#    pickle.dump([varSOM, dataArrays_orig, dataArrays, dataArray_train, map_lats, map_lons, months_doubles_allvars, years_doubles_allvars,
#                 year_month_strs_allvars, year_month_counts_allvars, mapedges, myfiles, varnames, SOMsize, mapSize, myregion], f)
    #pickle.dump([varnames, myfiles, dataArrays_orig, myregion, gridpoint_percentile_90, gridpoint_percentile_99, gridpoint_percentile_99_9, gridpoint_percentile_99_99,
#    pickle.dump([varnames, myfiles, myregion, percentile_90, percentile_95, percentile_99, percentile_99_5, percentile_99_9, percentile_99_99,
#                 indices_5mm_allvars, indices_10mm_allvars, indices_25mm_allvars, indices_50mm_allvars, indices_100mm_allvars, indices_150mm_allvars, indices_200mm_allvars, indices_250mm_allvars,
#                 indices_90th_allvars, indices_95th_allvars, indices_99th_allvars, indices_99_5th_allvars, indices_99_9th_allvars, indices_99_99th_allvars,
#                 datetimes_5mm_allvars, datetimes_10mm_allvars, datetimes_25mm_allvars, datetimes_50mm_allvars, datetimes_100mm_allvars, datetimes_150mm_allvars, datetimes_200mm_allvars, datetimes_250mm_allvars,
#                 datetimes_90th_allvars, datetimes_95th_allvars, datetimes_99th_allvars, datetimes_99_5th_allvars, datetimes_99_9th_allvars, datetimes_99_99th_allvars,
#                 count_5mm_allvars, count_10mm_allvars, count_25mm_allvars, count_50mm_allvars, count_100mm_allvars, count_150mm_allvars, count_200mm_allvars, count_250mm_allvars,
#                 count_90th_allvars, count_95th_allvars, count_99th_allvars, count_99_5th_allvars, count_99_9th_allvars, count_99_99th_allvars], f)
#    pickle.dump([varnames, myfiles, myregion, percentile_99_allvars, indices_99th_allvars, datetimes_99th_allvars, count_99th_allvars], f)
    pickle.dump([varnames, myfiles, myregion, indices_ari_allvars, datetimes_ari_allvars, count_ari_allvars], f)
#        # To open the file later
#        # with open('objs.pkl', 'rb') as f:  # Python 2: open(..., 'r') Python 2: open(..., 'rb')
#        #     obj0, obj1, obj2, ... = pickle.load(f)
#
##    return varSOM, dataArray, varSOM_nonzero, dataArray_nonzero, map_lats, map_lons, months_doubles, years_doubles, \
##    return varSOM, dataArray, map_lats, map_lons, months_doubles, years_doubles, \
##        year_month_strs, year_month_counts, mapedges
##    return varSOMs, dataArray_train_allSOMs, dataArrays_allSOMs, map_lats, map_lons
##    return varSOM, dataArray_train, dataArrays, map_lats, map_lons
#
#
##SOM_counter = -1
##for SOMsize in SOMsizes:
##    print(SOMsize)
##    SOM_counter = SOM_counter + 1
#SOMsize_str = str(SOMsize[0]) + 'x' + str(SOMsize[1])  # SOM grid size expressed as a string (e.g. '4x3')
##
##    varSOM = varSOMs[SOM_counter]
##
##    print('len(dataArray_train_allSOMs) = ' + str(len(dataArray_train_allSOMs[SOM_counter])))
##    winning_nodes = varSOM.win_map(dataArray_train_allSOMs[SOM_counter], return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
#winning_nodes = varSOM.win_map(dataArray_train, return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
##    winning_nodes = varSOM.win_map(dataArray_train_allSOMs[0], return_indices=True)  # for each SOM node, get the indices of the output times for which that node is the 'winner'
#                                                                                     # note that the SOM node indices are [col, row], and I will read through cols first
#print('winning_nodes:')
#print(winning_nodes)
#for myvar in range(0, len(mypaths)):
#    print('Creating ' + SOMsize_str + ' SOM figures for ' + varname_titles[myvar] + ' ' + myregion + '...')
#    fig = plt.figure(figsize=(12, 6))
#    node_count = -1
#    my_var_cmap = 'Blues'
#    my_var_vmin = 0
#    if myvar == 0:
#        my_var_vmax = 1#5#25#5
#        my_var_levels = [0,0.2,0.4,0.6,0.8,1]#[0,1,2,3,4,5]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
#        my_var_tick_levels = [0,0.2,0.4,0.6,0.8,1]#[0,1,2,3,4,5]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
#    elif myvar == 1:
#        my_var_vmax = 20#10#25#5
#        my_var_levels = [0,4,8,12,16,20]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
#        my_var_tick_levels = [0,4,8,12,16,20]#[0,2,4,6,8,10]#[0,5,10,15,20,25]#[0,1,2,3,4,5]
#    my_var_cbarextend = 'max'
#    my_var_cbarlabel = 'liquid precipitation [mm]'
#    for iSOM in range(0, SOMsize[1]):
#        for jSOM in range(0, SOMsize[0]):
#            winning_node_indices = winning_nodes[jSOM, iSOM]
#            print(str(SOMsize) + ': jSOM (SOMsize[0]) = ' + str(jSOM) + ', iSOM (SOMsize[1]) = ' + str(iSOM))
##                print('SOM_counter = ' + str(SOM_counter))
#            print('myvar = ' + str(myvar))
#            print('winning_node_indices:')
#            print(winning_node_indices)
#            node_count = node_count + 1
##                if myvar == 0:
##                print('dataArrays_allSOMs:')
##                print(dataArrays_allSOMs)
##                print('NaN count for dataArrays_allSOMs = ' + str(np.count_nonzero(np.isnan(dataArrays_allSOMs))))
##                print(dataArrays_allSOMs[0])
##                print('NaN count for dataArrays_allSOMs[0] = ' + str(np.count_nonzero(np.isnan(dataArrays_allSOMs[0]))))
##                print(dataArrays_allSOMs[0][0])
##                print('NaN count for dataArrays_allSOMs[0][0] = ' + str(np.count_nonzero(np.isnan(dataArrays_allSOMs[0][0]))))
##                print(dataArrays_allSOMs[0][0][0])
##                print('NaN count for dataArrays_allSOMs[0][0][0] = ' + str(np.count_nonzero(np.isnan(dataArrays_allSOMs[0][0][0]))))
##            print('dataArrays_allSOMs dimensions:')
##            print(len(dataArrays_allSOMs))  # 7 (w/ allSOMs)
##            print(len(dataArrays_allSOMs[0]))  # 2 (w/ allSOMs)
##            print(len(dataArrays_allSOMs[0][0]))  # 215 (w/ allSOMs)
##                print(len(dataArrays_allSOMs[0][0][0]))  # 61250 (w/ allSOMs)
#                    #dataArray_node = dataArrays_allSOMs[SOM_counter][myvar][0:mapSize_1D, winning_node_indices]
##                    dataArray_node = dataArrays_allSOMs[SOM_counter][myvar][winning_node_indices, 0:mapSize_1D]
##                dataArray_node = dataArrays_allSOMs[SOM_counter][myvar][winning_node_indices]
#            dataArray_node = dataArrays[myvar][winning_node_indices]
##                elif myvar == 1:
##                    print('dataArrays_allSOMs dimensions:')
##                    print(len(dataArrays_allSOMs))  # 7
##                    print(len(dataArrays_allSOMs[0]))  # 2
##                    print(len(dataArrays_allSOMs[0][0]))  # 215
##                    print(len(dataArrays_allSOMs[0][0][0]))  # 61250
##                    #dataArray_node = dataArrays_allSOMs[SOM_counter][myvar][mapSize_1D+1:2*mapSize_1D, winning_node_indices]
##                    dataArray_node = dataArrays_allSOMs[SOM_counter][myvar][winning_node_indices, mapSize_1D+1:2*mapSize_1D]
##                    dataArray_node = dataArrays_allSOMs[SOM_counter][myvar][winning_node_indices]
##                print('Printing dataArray_node')
##                print(dataArray_node)
#            print('dataArray_node.shape = ' + str(dataArray_node.shape))  # 86, 61250 in example
#            dataArray_node_map = np.mean(dataArray_node.reshape(-1, mapSize[1], mapSize[0]), axis=0)  # -1 --> value is inferred from the length of the array and remaining dimensions (keeps 1st dim len consistent if 2nd dim len = mapSize[1] * mapSize[0])
##                print('Printing dataArray_node_map for ' + SOMsize_str + ' node ' + str(node_count))
##                print(dataArray_node_map)
#            print('dataArray_node_map.shape = ' + str(dataArray_node_map.shape))
#            # Adjust the number and arrangement of the subplots based on the SOM grid size
#            if SOMsize_str == '3x2':
#                my_subplot = 231
#            elif SOMsize_str == '4x2':
#                my_subplot = 241
#            elif SOMsize_str == '4x3':
#                my_subplot = 341
#            elif SOMsize_str == '5x4':
#                my_subplot = 451
#            elif SOMsize_str == '2x2':
#                my_subplot = 221
#            elif SOMsize_str == '3x3':
#                my_subplot = 331
#            elif SOMsize_str == '4x4':
#                my_subplot = 441
#            elif SOMsize_str == '5x5':
#                my_subplot = 551
#                # cart_proj = ccrs.LambertConformal()
#            cart_proj = ccrs.PlateCarree()
#            ax = plt.subplot(SOMsize[1], SOMsize[0], node_count + 1, projection=cart_proj)
##                ax = plt.subplot(SOMsize[1], SOMsize[0], node_count, projection=cart_proj)
##                varSOM_plot = varSOM.get_weights()[jSOM-1, iSOM-1, :].reshape(mapSize[1], mapSize[0])
##                print(varSOM.get_weights()[jSOM-1, iSOM-1, :])
##                print('NaN count for varSOM weights at current node = ' + str(np.count_nonzero(np.isnan(varSOM.get_weights()[jSOM-1, iSOM-1, :]))))
##                ax_ = ax.contourf(map_lons, map_lats, np.transpose(varSOM_plot), cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
#            ax_ = ax.contourf(map_lons, map_lats, np.transpose(dataArray_node_map), cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
#                                              levels=my_var_levels, transform=ccrs.PlateCarree(), extend=my_var_cbarextend)
#            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=1.00, linestyle='--')
#            gl.xlabels_top = False
#            gl.xlabel_style = {'size': 6, 'color': 'black'}
#            gl.ylabels_left = False
#            gl.ylabel_style = {'size': 6, 'color': 'black'}
#            ax.coastlines('10m', linewidth=0.8)
#            country_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land',
#                                                           scale='10m', facecolor='none')
#            state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
#                                                         scale='10m', facecolor='none')
#            ax.add_feature(country_borders, edgecolor='black')
#            ax.add_feature(state_borders, edgecolor='black', linewidth=0.5)
#            ax.title.set_text('Node ' + str(node_count + 1))
#                # ax.outline_patch.set_alpha(0)  # plot outline/gridbox
#                # ax.background_patch.set_alpha(0)  # plot background
#                # plt.imshow(varSOM_plot, interpolation='none', aspect='auto')
#    cax = plt.axes([0.2, 0.525, 0.625, 0.025])
#    cbar = fig.colorbar(ax_, cax=cax, orientation='horizontal', extend=my_var_cbarextend,
#                        ticks=my_var_tick_levels, label=my_var_cbarlabel)#, pad=0.05)
#    plt.suptitle(varname_titles[myvar] + ' -- ' + date_first_str + '-' + date_last_str, x=0.5125, y=0.9125, fontsize=14)
#    plt.savefig('SOM_01H24Htrain_' + varnames[myvar] + '_' + SOMsize_str + '_learningrate25_' + date_first_str + '_' + date_last_str + '_' + myregion + '_test.png')#, bbox_inches='tight', pad_inches=0)
#
