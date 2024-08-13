# To run this script, follow these steps:
# - ncar_pylib my_ncar_pylib_clone (or my_ncar_pylib_clone_casper)
# - conda activate CFGRIB
# If new environment:
# - conda install cartopy
# - conda install -c conda-forge wrf-python
# - pip --no-cache-dir install mat73

# Import necessary modules

import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from wrf import (getvar, get_cartopy, cartopy_xlim, cartopy_ylim)
import xarray as xr
from netCDF4 import Dataset
import mat73
#import matlab.engine
#eng = matlab.engine.start_matlab()
import warnings
#warnings.filterwarnings("ignore", message="The outline_patch property is deprecated. ")
#warnings.filterwarnings("ignore", message="The background_patch property is deprecated. ")
#warnings.filterwarnings("ignore", message="The 'extend' parameter to Colorbar has no effect ")
#warnings.filterwarnings("ignore", message="/glade/u/home/naegele/miniconda3/envs/CFGRIB/")
#warnings.filterwarnings("ignore", message="ShapelyDeprecationWarning")
#def fxn():
#    warnings.warn("deprecated", DeprecationWarning)
#
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    fxn()
warnings.filterwarnings("ignore") # Not ideal, but other methods to ignore only Shapely deprecation warnings did not work

# User-defined variables are here
SOMsize = [3,2]#[4,3]#[3,2]#[4,3]#[5, 4]#[4, 3]#[4, 2]#[3, 2]
my_somsize = '3x2'#'4x3'#'3x2'#'4x3'#'5x4'#'4x3'#'4x2'#'3x2'
var_list = [['wspd80m', 'ght_850', 'ght_500', 'hfx', 'pblh', 'ust', 'power_density'],
            ['wspd80m', 'mslma', 'gh', 'pt', 'fric_v', 'shtfl', 'lhtfl', 'wspd10m80mdiff', 'wspddiff10m', 'wspddiff80m']]
           #  'wspd80m4x2', 'wspd80m4x3']]
           # ['wspd80m', 'pt', 'shtfl', 'mslma', 'gh', '10_80_diff']]
var_name_list = [['SOMs trained on WRF 80-m wind components', 'WRF 850-hPa geopotential height',
                  'WRF 500-hPa geopotential height', 'WRF surface heat flux', 'WRF PBL depth', 
                  'WRF frictional velocity ($u_*$)', 'WRF power density'], ['SOMs trained on HRRR 80-m wind components', 
                  'HRRR mean sea level pressure', 'HRRR 500-hPa geopotential height', 'HRRR 2-m potential temperature', 
                  'HRRR frictional velocity ($u_*$)', 'HRRR sensible heat flux', 'HRRR latent heat flux',
                  'HRRR 10-80-m wind component difference', 'HRRR 10-m wind', 'HRRR 80-m wind']]
                 # 'SOMs trained on HRRR 80-m wind components', 'SOMs trained on HRRR 80-m wind components']]
var_cmap_list = [['Purples', 'Greens', 'Greens', 'YlOrBr', 'YlGn', 'Oranges', 'Blues'], ['Purples', 'BuPu', 'Greens', 'turbo',
                  'Oranges', 'YlOrBr', 'YlGnBu', 'Purples', 'Purples', 'Purples']]#, 'Purples', 'Purples']]
var_vmin_list = [[0, 1400, 5500, 0, 0, 0, 0], [0, 990, 5200, 270, 0, 0, 0, 0, 0, 0]]#, 0, 0]]
var_vmax_list = [[10, 1550, 5900, 200, 3000, 0.5, 1000], [12, 1020, 5800, 300, 0.5, 100, 100, 5, 12, 12]]#, 10, 10]]
var_levels_list = [[[0,3,4,5,6,7,8,9,10], [1400,1410,1420,1430,1440,1450,1460,1470,1480,1490,1500,1510,1520,1530,1540,1550],
                    np.arange(5500,5900+10,10), [0,25,50,75,100,125,150,175,200], 
                    [0,250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000], [0,0.1,0.2,0.3,0.4,0.5],
                    [0,100,200,300,400,500,600,700,800,900,1000]], 
                   [[0,3,4,5,6,7,8,9,10,11,12], np.arange(990,1020+1,5), 
                    np.arange(5200,5800+50,50), np.arange(270,300+1,2), 
                    [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5], [0,10,20,40,60,80,100], [0,20,40,60,80,100], 
                    [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5], [0,3,4,5,6,7,8,9,10,11,12], [0,3,4,5,6,7,8,9,10,11,12]]]
                   # [0,3,4,5,6,7,8,9,10], [0,3,4,5,6,7,8,9,10]]]
var_tick_levels_list = [[[0,3,4,5,6,7,8,9,10], [1400,1450,1500,1550], [5500,5600,5700,5800,5900], [0,50,100,150,200], 
                         [0,500,1000,1500,2000,2500,3000], [0,0.1,0.2,0.3,0.4,0.5], [0,100,200,300,400,500,600,700,800,900,1000]], 
                        [[0,3,4,5,6,7,8,9,10,11,12], 
                         [990,1000,1010,1020], [5200,5300,5400,5500,5600,5700,5800], 
                         [270,280,290,300], [0,0.1,0.2,0.3,0.4,0.5], [0,10,20,40,60,80,100], [0,20,40,60,80,100], 
                         [0,1,2,3,4,5], [0,3,4,5,6,7,8,9,10,11,12], [0,3,4,5,6,7,8,9,10,11,12]]]
                       #  [0,3,4,5,6,7,8,9,10], [0,3,4,5,6,7,8,9,10]]]
var_cbarextend_list = [['max', 'both', 'both', 'both', 'max', 'max', 'max'], ['max', 'both', 'both', 'both', 'max', 'both', 'both',
                        'max', 'max', 'max']]#, 'max', 'max']]
var_cbarlabel_list = [['wind speed [m s$^{-1}$]', 'geopotential height [m]', 'geopotential height [m]', 
                       'surface heat flux [W m$^{-2}$]', 'PBL depth [m]', 'frictional velocity [m s$^{-1}$]', 'power density [W m$^{-2}$]'], 
                      ['wind speed [m s$^{-1}$]', 'sea level pressure [hPa]', 'geopotential height [m]', 
                       'potential temperature [K]', 'frictional velocity [m s$^{-1}$]', 'sensible heat flux [W m$^{-2}$]', 
                       'latent heat flux [W m$^{-2}$]', 'wind component difference [m s$^{-1}$]', 'wind speed [m s$^{-1}$]',
                       'wind speed [m s$^{-1}$]']]#, 'wind speed [m s$^{-1}$]', 'wind speed [m s$^{-1}$]']]

# This is the directory containing the WRF output, and can be edited if needed

myfile_wrf_wind = '/glade/work/naegele/wrf_soms_wspd78m_output3.mat'
myfile_wrf_myvar = '/glade/work/naegele/wrf_soms_wspd78m_myvar_output.mat'
myfile_wrf_powerdensity = '/glade/u/home/naegele/tims_som_code/power_density_wrf_v73.mat' # 3x2 grid
myfile_wrf_latlon = '/glade/campaign/ral/wsap/kuwait/wrf/exp9/2017-09-01_00/wrfout_solarwind_d01_2017-09-01_00:00:00'
myfile_hrrr_wind = '/glade/work/naegele/hrrr_soms_output.mat'
#myfile_hrrr_winddiff = '/glade/work/naegele/hrrr_soms_wspd10m80mshear_output.mat'
myfile_hrrr_winddiff = '/glade/work/naegele/hrrr_soms_wspd10m80mshear_output_postdefense.mat'
#myfile_hrrr_winddiff10m = '/glade/work/naegele/hrrr_soms_wspd10m80mdifftrain_wspd10m_output.mat'
#myfile_hrrr_winddiff80m = '/glade/work/naegele/hrrr_soms_wspd10m80mdifftrain_wspd80m_output.mat'
myfile_hrrr_winddiff10m = '/glade/work/naegele/hrrr_soms_wspd10m80mdifftrain_wspd10m_output_postdefense.mat'
myfile_hrrr_winddiff80m = '/glade/work/naegele/hrrr_soms_wspd10m80mdifftrain_wspd80m_output_postdefense.mat'
myfile_hrrr_wind4x2 = '/glade/work/naegele/hrrr_soms_output_diffsize_4x2.mat' # y is 8x38160
myfile_hrrr_wind4x3 = '/glade/work/naegele/hrrr_soms_output_diffsize.mat' # y is 12x38160
myfile_hrrr_wind5x4 = '/glade/work/naegele/hrrr_soms_wspd80m_5x4_output_postdefense.mat'
#myfile_hrrr_myvar = '/glade/work/naegele/hrrr_soms_wspd80m_wspd10m80mdiff_myvar_output.mat'
myfile_hrrr_myvar = '/glade/work/naegele/hrrr_soms_wspd80m_wspd10m80mdiff_myvar_output_postdefense.mat'
myfile_hrrr_mslma = '/glade/work/naegele/mslp_test_fixed_full.mat'
myfile_hrrr_ght_500 = '/glade/work/naegele/ght_500_test_fixed_full.mat'
#myfile_hrrr_latlon = '/glade/scratch/naegele/hrrr_data/test_hrrr_data/hrrr/20211231/subset_20211231_hrrr.t00z.wrfsfcf03.grib2'
myfile_hrrr_latlon = '/glade/scratch/naegele/hrrr_data/test_hrrr_data/hrrr.t00z.wrfsfcf03.nc'


model_list = ['wrf','hrrr']
for my_model_ind in range(0,0+1):#1, len(var_list)):
    my_model = model_list[my_model_ind]
    if my_model == 'wrf':
        # Read in lat-lon data
        dataset_wrf_latlon = Dataset(myfile_wrf_latlon)
        XLAT = getvar(dataset_wrf_latlon, 'XLAT')
        XLONG = getvar(dataset_wrf_latlon, 'XLONG')
        XLAT = XLAT[::5,0:300:5]  # only get every 5th element, and cut off rightmost 100 points
        XLONG = XLONG[::5,0:300:5]  # only get every 5th element
    elif my_model == 'hrrr':
        #ds = xr.open_dataset(myfile_hrrr_latlon, engine='cfgrib')
        #latitude = ds.coords['latitude']
        #longitude = ds.coords['longitude']
        #latitude = latitude[530:1060:5,1440:1800:2]
        #longitude = longitude[530:1060:5,1440:1800:2]
        nc_fid = Dataset(myfile_hrrr_latlon, 'r')
        #latitude = nc_fid.variables['gridlat_0'][:]
        lats = nc_fid.variables['gridlat_0'][530:1060:5,1440:1800:2]
        lons = nc_fid.variables['gridlon_0'][530:1060:5,1440:1800:2]
    big_counter = -1
    #for my_var_ind in range(0, len(var_list_wrf)):
    for my_var_ind in range(0,5+1):#(2,2+1):# len(var_list[my_model_ind])):
        #my_var_ind = 2  # index for variable, variable name, and other variable-specific items
        my_var = var_list[my_model_ind][my_var_ind]
        my_var_name = var_name_list[my_model_ind][my_var_ind]
        my_var_cmap = var_cmap_list[my_model_ind][my_var_ind]
        my_var_vmin = var_vmin_list[my_model_ind][my_var_ind]
        my_var_vmax = var_vmax_list[my_model_ind][my_var_ind]
        my_var_levels = var_levels_list[my_model_ind][my_var_ind]
        my_var_tick_levels = var_tick_levels_list[my_model_ind][my_var_ind]
        my_var_cbarextend = var_cbarextend_list[my_model_ind][my_var_ind]
        my_var_cbarlabel = var_cbarlabel_list[my_model_ind][my_var_ind]

        # Retrieve the variable dataset from its file
        if my_model == 'wrf':
            if my_var == 'wspd80m':
                print('Accessing WRF output from: ' + myfile_wrf_wind)
                dataset = mat73.loadmat(myfile_wrf_wind)  # wind data only -- loads as a dictionary
            elif my_var == 'power_density':
                print('Accessing WRF output from: ' + myfile_wrf_powerdensity)
                dataset = mat73.loadmat(myfile_wrf_powerdensity)  # wind data only -- loads as a dictionary
            else:
                print('Accessing WRF output from: ' + myfile_wrf_myvar)
                dataset = mat73.loadmat(myfile_wrf_myvar) # all vars besides wind
                # print(dataset.variables)
        elif my_model == 'hrrr':
            if my_var == 'wspd80m':
                if my_somsize == '3x2':
                    print('Accessing HRRR output from: ' + myfile_hrrr_wind)
                    dataset = mat73.loadmat(myfile_hrrr_wind)  # wind data only -- loads as a dictionary
                elif my_somsize == '4x2':
                    print('Accessing HRRR output from: ' + myfile_hrrr_wind4x2)
                    dataset = mat73.loadmat(myfile_hrrr_wind4x2)  # wind data only -- loads as a dictionary
                elif my_somsize == '4x3':
                    print('Accessing HRRR output from: ' + myfile_hrrr_wind4x3)
                    dataset = mat73.loadmat(myfile_hrrr_wind4x3)  # wind data only -- loads as a dictionary
                elif my_somsize == '5x4':
                    print('Accessing HRRR output from: ' + myfile_hrrr_wind5x4)
                    dataset = mat73.loadmat(myfile_hrrr_wind5x4)  
            elif my_var == 'wspd10m80mdiff':
                print('Accessing HRRR output from: ' + myfile_hrrr_winddiff)
                dataset = mat73.loadmat(myfile_hrrr_winddiff)  # wind diff data only -- loads as a dictionary
            elif my_var == 'wspddiff10m':
                print('Accessing HRRR output from: ' + myfile_hrrr_winddiff10m)
                dataset = mat73.loadmat(myfile_hrrr_winddiff10m)  # wind diff data only -- loads as a dictionary
            elif my_var == 'wspddiff80m':
                print('Accessing HRRR output from: ' + myfile_hrrr_winddiff80m)
                dataset = mat73.loadmat(myfile_hrrr_winddiff80m)  # wind diff data only -- loads as a dictionary
            elif my_var == 'mslma':
                print('Accessing HRRR output from: ' + myfile_hrrr_mslma)
                dataset = mat73.loadmat(myfile_hrrr_mslma)  # wind diff data only -- loads as a dictionary
            elif my_var == 'gh':
                print('Accessing HRRR output from: ' + myfile_hrrr_ght_500)
                dataset = mat73.loadmat(myfile_hrrr_ght_500)  # wind diff data only -- loads as a dictionary
            else:
                print('Accessing HRRR output from: ' + myfile_hrrr_myvar)
                dataset = mat73.loadmat(myfile_hrrr_myvar) # vars: mslma, gh, pt, fric_v, shtfl, lhtfl

        # Create a figure
        print('Making figure for ' + my_model + ' ' + my_var)
        fig = plt.figure(figsize=(6,6))
        if my_var == 'wspd80m' or my_var == 'wspd10m80mdiff':
            #print(dataset.keys())
            # dataArray = dataset['dataArray']  # 7200x17520
            net = dataset['net']
            y = net['IW']
        elif my_var == 'wspddiff10m' or my_var == 'wspddiff80m':
            wspddiff_1level = dataset['all_wspd_nodeavg'] # 2-D
            Uwspddiff_1level = dataset['all_Uwspd_nodeavg'] # 1-D
            Vwspddiff_1level = dataset['all_Vwspd_nodeavg'] # 1-D
        elif my_var == 'power_density':
            pd_node1avg = dataset['powerdensity_node1avg']
            pd_node2avg = dataset['powerdensity_node2avg']
            pd_node3avg = dataset['powerdensity_node3avg']
            pd_node4avg = dataset['powerdensity_node4avg']
            pd_node5avg = dataset['powerdensity_node5avg']
            pd_node6avg = dataset['powerdensity_node6avg']
            powerdensity_nodeavgs = [pd_node1avg, pd_node2avg, pd_node3avg, pd_node4avg, pd_node5avg, pd_node6avg]
        else:
            all_var_nodeavg = dataset['all_var_nodeavg'] 
            # wrf - 60x1800 = mapSize(0) x mapSize(1)*SOMsize(0)*SOMsize(1)*(length(var_list_wrf)-1)
            # hrrr - 180x3816 = 180 x 106*3*2*6
        counter = -1
        for iSOM in range(1, SOMsize[1]+1):
            for jSOM in range(1, SOMsize[0]+1):
                counter = counter + 1
                # Adjust the number and arrangement of the subplots based on the SOM grid size
                if my_somsize == '3x2':
                    my_subplot = 231
                elif my_somsize == '4x2':
                    my_subplot = 241
                elif my_somsize == '4x3':
                    my_subplot = 341
                elif my_somsize == '5x4':
                    my_subplot = 451

                # Get the cartopy mapping object and assign the map size, latitude, and longitude
                if my_model == 'wrf':                    
                    cart_proj = get_cartopy(XLAT)
                    map_size = [60, 60]
                    map_lat = XLAT
                    map_lon = XLONG
                elif my_model == 'hrrr':
                    #cart_proj = get_cartopy(latitude)
                    cart_proj = ccrs.LambertConformal()
                    #cart_proj = ccrs.PlateCarree()
                    map_size = [106, 180] #[180, 106] # actual map is 106x180
                    map_lat = lats
                    map_lon = lons

                if my_var == 'wspd80m' or my_var == 'wspd10m80mdiff':
                    thisSOM = y[0][counter][range(0, map_size[0]*map_size[1])]
                    thatSOM = y[0][counter][range(map_size[0]*map_size[1], 2*map_size[0]*map_size[1])]
                    thisSOM = thisSOM.reshape(map_size[0], map_size[1])
                    thatSOM = thatSOM.reshape(map_size[0], map_size[1])
                    speedSOM = (thisSOM**2 + thatSOM**2)**0.5

                    #ax = plt.subplot(my_subplot+counter, projection=cart_proj)
                    #ax = plt.subplot(3,4,counter+1,projection=cart_proj)
                    ax = plt.subplot(2,3,counter+1,projection=cart_proj)
                    #ax = plt.subplot(4,5,counter+1,projection=cart_proj)
                    #if my_model == 'hrrr':
                    #    speedSOM = np.transpose(speedSOM)
                    ax_ = ax.contourf(map_lon, map_lat, speedSOM, cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
                                      levels=my_var_levels, transform=ccrs.PlateCarree(), extend=my_var_cbarextend)
                elif my_var == 'wspddiff10m' or my_var == 'wspddiff80m':
                    thisSOM = Uwspddiff_1level[counter]
                    thatSOM = Vwspddiff_1level[counter]
                    thisSOM = thisSOM.reshape(map_size[0], map_size[1])
                    thatSOM = thatSOM.reshape(map_size[0], map_size[1])
                    speedSOM = (thisSOM**2 + thatSOM**2)**0.5
                    #ax = plt.subplot(my_subplot+counter, projection=cart_proj) # Gives "IndexError: GridSpec slice would result in no space allocated for subplot"
                    ax = plt.subplot(3,4,counter+1,projection=cart_proj)
                    ax_ = ax.contourf(map_lon, map_lat, speedSOM, cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
                                      levels=my_var_levels, transform=ccrs.PlateCarree(), extend=my_var_cbarextend)
                elif my_var == 'power_density':
                    pd = np.transpose(powerdensity_nodeavgs[counter])
                    ax = plt.subplot(my_subplot+counter, projection=cart_proj)
                    ax_ = ax.contourf(map_lon, map_lat, pd, cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
                                      levels=my_var_levels, transform=ccrs.PlateCarree(), extend=my_var_cbarextend)
                else:
                    big_counter = big_counter + 1
                    # myvar = all_var_nodeavg[my_var_ind-1]
                    if my_var != 'mslma' and my_var != 'gh':
                        myrange = np.arange(0+(map_size[0]*big_counter), map_size[0]+(map_size[0]*big_counter))
                        #print(myrange)
                        #print(len(all_var_nodeavg))
                        #print(len(all_var_nodeavg[0]))
                        #myvar = np.transpose(all_var_nodeavg[:,myrange])
                        myvar = all_var_nodeavg[:,myrange]
                        #print(len(myvar))
                        #print(len(myvar[0]))
                    elif my_var == 'mslma' or my_var == 'gh':
                        ### all_var_nodeavg for MSLP is 180x1272, or 180x(106*12)
                        #print(len(all_var_nodeavg)) # 180
                        #print(len(all_var_nodeavg[0])) # 1272
                        #print(counter)
                        #print(0+(map_size[0]*counter))
                        #print(map_size[0]+(map_size[0]*counter))
                        myrange = np.arange(0+(map_size[0]*counter), map_size[0]+(map_size[0]*counter))
                        myvar = all_var_nodeavg[:,myrange]
                        #print(myvar)
                    #ax = plt.subplot(my_subplot+counter, projection=cart_proj)
                    #ax = plt.subplot(3,4,counter+1,projection=cart_proj)
                    ax = plt.subplot(2,3,counter+1,projection=cart_proj)
                    #if my_model == 'hrrr':
                    myvar = np.transpose(myvar)
                    if my_var == 'mslma':
                       myvar = myvar*(1/100) # convert from Pa to hPa
                    ax_ = ax.contourf(map_lon, map_lat, myvar, cmap=my_var_cmap, vmin=my_var_vmin, vmax=my_var_vmax,
                                      levels=my_var_levels, transform=ccrs.PlateCarree(), extend=my_var_cbarextend)
                if my_model == 'wrf':
                    ax.set_xlim(cartopy_xlim(map_lat))
                    ax.set_ylim(cartopy_ylim(map_lat))
                    plt.scatter(x=47.05, y=29.20, s=25, marker='*', facecolors='none', edgecolors='k', zorder=5,
                                transform=ccrs.PlateCarree())
                elif my_model == 'hrrr':
                    plt.scatter(x=287.28, y=39.97, s=25, marker='*', facecolors='none', edgecolors='k', zorder=5,
                                transform=ccrs.PlateCarree())
                    plt.scatter(x=286.57, y=39.55, s=25, marker='*', facecolors='none', edgecolors='k', zorder=5,
                                transform=ccrs.PlateCarree())
                if my_var == 'wspd80m' or my_var == 'wspd10m80mdiff' or my_var == 'wspddiff10m' or my_var == 'wspddiff80m':
                    X, Y = np.mgrid[0:map_size[0], 0:map_size[1]]
                    #X, Y = np.mgrid[map_lon.min():map_lon.max(), map_lat.min():map_lat.max()]
                    #print(X)
                    #print(len(X))
                    #print(len(X[0]))
                    #print(Y)
                    #print(len(Y))
                    #print(len(Y[0]))
                    #print(thisSOM)
                    #print(len(thisSOM))
                    #print(len(thisSOM[0]))
                    #print(thatSOM)
                    #print(len(thatSOM))
                    #print(len(thatSOM[0]))
                    lw = 2 * speedSOM / speedSOM.max()
                    #print(lw)
                    #    lw = np.transpose(lw)
                    #print(len(lw))
                    #print(len(lw[0]))
                    #if my_model == 'hrrr':
                    #    cart_proj = ccrs.PlateCarree()
                    #    hold on
                    #    bx = plt.subplot(231+counter, projection=cart_proj)
                    if my_model == 'wrf':
                        ax.streamplot(X, Y, thisSOM, thatSOM, density=0.4, color='r', linewidth=lw, 
                                      transform=ccrs.PlateCarree())
                                      #transform=ccrs.LambertConformal())
                    elif my_model == 'hrrr':
                        ax.streamplot(map_lon, map_lat, thisSOM, thatSOM, density=0.3, color='r', linewidth=lw, 
                                      transform=ccrs.PlateCarree())
                # op_som = np.transpose(eng.vec2ind(net[dataArray]))
                # wspd_80m_indices = np.where(op_som == counter)  # get how much data is represented by node
                # ax.title.set_text(['Node ', counter, '(', len(wspd_80m_indices)])
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=1.00, linestyle='--')
                gl.xlabels_top = False
                gl.xlabel_style = {'size': 6, 'color': 'black'}
                gl.ylabels_left = False
                gl.ylabel_style = {'size': 6, 'color': 'black'}
                ax.coastlines('10m', linewidth=0.8)
                country_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land',
                                                               scale='10m', facecolor='none')
                ax.add_feature(country_borders, edgecolor='black')
                ax.title.set_text('Node ' + str(counter+1))
                ax.outline_patch.set_alpha(0)  # plot outline/gridbox
                ax.background_patch.set_alpha(0)  # plot background

        if my_model == 'wrf':
            cax = plt.axes([0.2, 0.525, 0.625, 0.025])
        elif my_model == 'hrrr':
            if my_somsize == '4x3' or my_somsize == '5x4':
                cax = plt.axes([0.2, 0.1, 0.625, 0.025])
            else:
                cax = plt.axes([0.2, 0.5, 0.625, 0.025])
        cbar = fig.colorbar(ax_, cax=cax, orientation='horizontal', extend=my_var_cbarextend,
                            ticks=my_var_tick_levels, label=my_var_cbarlabel)#, pad=0.05)

        if my_model == 'wrf':
            plt.suptitle(my_var_name, x=0.5125, y=0.9125, fontsize=14)
        elif my_model == 'hrrr':
            plt.suptitle(my_var_name, x=0.5125, y=1.0000, fontsize=14)
            # Set the spacing between subplots
            if my_somsize == '4x3' or my_somsize == '5x4':
                plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
            else:
                plt.subplots_adjust(left=0.1, bottom=0.0, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

        # Save figure (and remove extra "white space" outside figure), and dpi=100 is default
        plt.savefig('SOM' + my_model + '_' + my_somsize + '_' + my_var + '_domain_full_highres_postdefense_v2.png')#, bbox_inches='tight', pad_inches=0)



