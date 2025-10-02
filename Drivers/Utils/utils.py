#####################
#
# to ChatGPT
# SO, again if I have defined a dict, how do I enable the dict.key vs dict['key'] functionality? 
#  ...
#  ... code below
# Your dictionary definitions
# my_dict = AttrDict({"name": "John", "age": 30})

"""
The first AttrDict implementation has a clear separation between the underlying dictionary data (_data) and the class's attributes. This separation can be useful in certain situations where you want to avoid potential conflicts between dictionary keys and class attributes. For example, if you have a dictionary key that matches a method or attribute name of the AttrDict class, the first implementation would prevent any conflicts, as the dictionary data is stored separately in the _data attribute.

However, for most practical purposes, the second implementation is more straightforward and convenient, as it allows direct access to the dictionary keys as attributes while still retaining all the functionality of a regular dictionary. It's also more concise and easier to understand for those who are familiar with Python's built-in dict class.

In summary, the first implementation provides a clear separation between class attributes and dictionary keys, which can be useful in certain edge cases, while the second implementation is generally more convenient and straightforward for everyday use.
"""
"""
# This doesn't allow both dict.key and dict['key'] syntax
class AttrDict:
    def __init__(self, data=None):
        if data is None:
            data = {}
        self.__dict__["_data"] = data

    def __getattr__(self, key):
        if key in self._data:
            return self._data[key]
        raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self._data[key] = value

    def __delattr__(self, key):
        if key in self._data:
            del self._data[key]
        else:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")
"""
# This allow both dict.key and dict['key'] syntax
class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")


def run(code):
    #import time
    print("exec(open(" +"'"+code+"'"+ ").read())")
    #print("will sleep in ",code)
    #time.sleep(10)

def env():
    import matplotlib.pyplot as plt

def HowManyWorkers(): 
    import os
    # set number of pools to create
    #nworkers = cpu_count()
    nworkers = len(os.sched_getaffinity(0))
    print('Flexible VertRegrid using reshaped ARRAYS ')
    print('nworkers available ',nworkers)

    return nworkers


def MakePath( user='juliob' , exp='YaYa', subd='hist', hsPat='cam.h0' , ymdPat='*' ,verbose=False):

    if (user in ['juliob','pel','tilmes'] ):
        path = f'/glade/derecho/scratch/{user}/archive/{exp}/atm/{subd}/{exp}.{hsPat}.{ymdPat}.nc' 
    elif (user == 'juliob_run' ):
        path = f'/glade/derecho/scratch/juliob/{exp}/run/{exp}.cam.h0.*.nc' 
    elif (user == 'juliob_camp' ):
        path = f'/glade/campaign/cgd/amp/juliob/archive/{exp}/atm/{subd}/{exp}.{hsPat}.{ymdPat}.nc'   
    elif (user == 'CMIP6' ):
        dir='/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/f.e21.FWHISTBgcCrop.f09_f09_mg17.CMIP6-AMIP-WACCM.001/atm/proc/tseries/month_1/'
        path=dir+ 'f.e21.FWHISTBgcCrop.f09_f09_mg17.CMIP6-AMIP-WACCM.001.cam.h0.U.195001-201412.nc'


    if (verbose==True):
        print( path )


    return path

def MakeDict4Exp( user='juliob' , exp='YaYa', subd='hist', hsPat='cam.h0' , ymdPat='*' ,
                 Src=None, Hkey=None, 
                 cmip_fld=None, ymdPat_is_range=False,
                 verbose=False, open_dataset=False, help=False, add_coords=False, shift_lons=False ):
    import xarray as xr
    import glob
    import numpy as np

    if (help == True):
        print( f" Possible 'users': 'juliob','pel','tilmes''juliob_run' 'juliob_camp' 'amwg_runs'  'omwg_mom6' 'CMIP6' 'CMIP6_WACCM' ")
        return
    
    if (user in ['juliob','pel','tilmes','hannay','bramberg','islas','gmarques',] ):
        bpath = f'/glade/derecho/scratch/{user}/archive/{exp}/atm/{subd}/'  #{exp}.{hsPat}.{ymdPat}.nc' 
    elif (user == 'juliob_run' ):
        bpath = f'/glade/derecho/scratch/juliob/{exp}/run/'  #{exp}.{hsPat}.{ymdPat}.nc' 
    elif (user == 'juliob_camp' ):
        bpath = f'/glade/campaign/cgd/amp/juliob/archive/{exp}/atm/{subd}/' # {exp}.{hsPat}.{ymdPat}.nc'   
    elif (user == 'amwg_runs' ):
        bpath = f'/glade/campaign/cgd/amp/amwg/runs/{exp}/atm/{subd}/' #{exp}.{hsPat}.{ymdPat}.nc'   
    elif (user == 'cesm_runs' ):
        bpath = f'/glade/campaign/cgd/amp/amwg/runs/{exp}/atm/{subd}/' #{exp}.{hsPat}.{ymdPat}.nc'   
    elif (user == 'omwg_mom6' ):
        bpath = f'/glade/campaign/cesm/development/omwg/projects/MOM6/' #{exp}/atm/{subd}/{exp}.{hsPat}.{ymdPat}.nc'   
    elif (user == 'CMIP6_WACCM' ):
        dir='/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/f.e21.FWHISTBgcCrop.f09_f09_mg17.CMIP6-AMIP-WACCM.001/atm/proc/tseries/month_1/'
        bpath=f'{dir}f.e21.FWHISTBgcCrop.f09_f09_mg17.CMIP6-AMIP-WACCM.001.cam.h0.{cmip_fld}.{ymdPat}.nc'
        PSpath=f'{dir}f.e21.FWHISTBgcCrop.f09_f09_mg17.CMIP6-AMIP-WACCM.001.cam.h0.PS.{ymdPat}.nc'
    elif (user == 'CMIP6' ):
        dir='/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/f.e21.FHIST_BGC.f09_f09_mg17.CMIP6-AMIP.001/atm/proc/tseries/month_1/'
        bpath=f'{dir}f.e21.FHIST_BGC.f09_f09_mg17.CMIP6-AMIP.001.cam.h0.{cmip_fld}.{ymdPat}.nc'
        PSpath=f'{dir}f.e21.FHIST_BGC.f09_f09_mg17.CMIP6-AMIP.001.cam.h0.PS.{ymdPat}.nc'

    if ( isinstance(ymdPat, list ) == False ):
        #print( f'Is ymdPat a list {isinstance(ymdPat, list )}' )
        if (user in ('CMIP6','CMIP6_WACCM')):
            path=bpath
        else:
            path=f'{bpath}{exp}.{hsPat}.{ymdPat}.nc'
    else:
        print( 'ymdPat is a list' )
        if (ymdPat_is_range==False):
            print( 'ymdPat is a list of year strings' )
            path=[]
            for ymdPat_ in ymdPat :
                newpath=sorted( glob.glob( f'{bpath}{exp}.{hsPat}.{ymdPat_}.nc' ) )
                #print(ymdPat_, newpath)
                path = path + newpath
        else:
            print( 'ymdPat is a 2 elem list with a range' )
            year0 = int(ymdPat[0])
            year1 = int(ymdPat[1])
            path=[]
            ipath=0
            for y in np.arange( start=year0, stop=year1+1 ):
                ymdPat_ = f'{str(y).zfill(4)}-*'
                newpath = sorted( glob.glob( f'{bpath}{exp}.{hsPat}.{ymdPat_}.nc' ) )
                path = path + newpath
                ipath=ipath+len(newpath)
            print( f'Range resulted in path with {ipath} members' )

    
    ###/glade/u/home/gmarques/campaign_MOM6/b.cesm3_cam058_mom_e.B1850WscMOM.ne30_L58_t061.camdev_cice5.026g
    ### /glade/campaign/cesm/development/omwg/projects/MOM6/

    if (verbose==True):
        if ( isinstance(ymdPat, list ) == False ):
            print( path )
        else:
            print( path[0], '\n', path[-1] )

    dex = { 'exp':exp ,
           'user':user, 
           'subd':subd,
           'hsPat':hsPat,
           'ymdPat':ymdPat,
           'path':path }
    
    if (Src is not None):
        dex['Src']=Src
    if (Hkey is not None):
        dex['Hkey']=Hkey
        
    
    if (open_dataset==True):
        X = xr.open_mfdataset( path ,data_vars='different', coords='different' )
        if (user in ('CMIP6','CMIP6_WACCM')):
            PS = xr.open_mfdataset( PSpath ,data_vars='different', coords='different' )
            X['PS'] = PS['PS']

        if (shift_lons==True):
            X['lon'] = xr.where(X['lon'] > 180, X['lon'] - 360, X['lon'])
            # Roll the dataset along the longitude axis
            X = X.roll(lon=len(X['lon']) // 2, roll_coords=True)

        dex['X']=X
        if (add_coords==True):
            lon=X.lon.values
            lat=X.lat.values
            lev=X.lev.values
            zlev=-7.0*np.log( lev/1_000. )
            dex['lon']  =lon
            dex['lat']  =lat
            dex['lev']  =lev
            dex['zlev'] =zlev

    # Use class defined above to enable 'dict.key' syntax.
    #--------------------------------
    Adex=AttrDict( dex )
    
    return Adex

def extValues( DX , flds ):

    dex = { 'exp':exp ,
           'user':user, 
           'subd':subd,
           'hsPat':hsPat,
           'ymdPat':ymdPat,
           'path':path }

    fldx = ['lat','lon','lev']+flds

    for fld in fldx:
        dex[ fld ] = DX.X[fld].values

    return dex

def trim_to_year( D , nyr_max=1000 ):
    import numpy as np
    
    nyr=np.minimum( nyr_max*12 , 12*(len(D.X.time)//12) )
    print(f"Discarding last {len(D.X.time)-nyr} months of {D.exp}" )
    D.X=D.X.isel(time=np.arange(nyr))


def days_in_month(year, month, check_for_leap_year=False):
    # February logic with optional leap year check
    if month == 2:
        if check_for_leap_year:
            print( "Adjusting days in February to 29 for leap years" )
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                return 29
            else:
                return 28
        else:
            print( "28 days in February. Not worried about leap years" )
            return 28
    # Months with 30 days
    elif month in [4, 6, 9, 11]:
        return 30
    # Months with 31 days
    elif month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    else:
        raise ValueError("Invalid month. Month should be an integer between 1 and 12.")

# Function to find the nearest index, handling out-of-range cases
def find_nearest_plev_indices(plev=None, target_levels=None):
    import numpy as np
    indices = []
    for level in target_levels:
        if level > np.max(plev):  # if above the highest level
            indices.append(-1)
        elif level < np.min(plev):  # if below the lowest level
            indices.append(-1)
        else:
            # Find the nearest index
            nearest_index = (np.abs(plev - level)).argmin()
            indices.append(nearest_index)

    indices = np.array( indices )
        
    return indices

################
def ymds(year,month=None,day=None,hour=None,append=None):

    if month is None:
        ymdPat = str(year).zfill(4)
    elif day is None:
        ymdPat = f'{str(year).zfill(4)}-{str(month).zfill(2)}'
    elif hour is None:
        ymdPat = f'{str(year).zfill(4)}-{str(month).zfill(2)}-{str(day).zfill(2)}'
    else:
        sec=3600*hour
        ymdPat = f'{str(year).zfill(4)}-{str(month).zfill(2)}-{str(day).zfill(2)}-{str(sec).zfill(5)}'

    if append is not None:
        ymdPat=f'{ymdPat}{append}'
       

    return ymdPat

################
def area_from_latlon(lat=None, lon=None):
    import numpy as np
    from Utils import constants as Co

    Re = Co.Rearth()
    Pi = Co.pi()

    nx=len(lon)
    ny=len(lat)
    # np.abs in case grids are ass-backwards like ERA
    dxr=np.abs( ( lon[1]-lon[0] ) )* (Pi /180. )
    dyr=np.abs( ( lat[1]-lat[0] ) )* (Pi /180. )
    Lon,Lat=np.meshgrid(lon,lat)
    Latr = Lat * (Pi /180. )
    area = ( Re **2 ) * dxr * dyr * np.cos( Latr )

    return area
    
    

