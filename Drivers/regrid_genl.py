#!/usr/bin/env python
################################################
# New style 
# ###############################################
import sys
import os


from Utils import GridUtils as GrU
from Utils import utils as uti

#from PyRegridding.Utils import MakePressures as MkP
#from Drivers import RegridField as RgF
import RegridField as RgF
import update_config as uc

# The usual
from datetime import date
import numpy as np
import xarray as xr

# Some other useful packages 
import copy
import time
import cftime
import yaml
import numbers

# Some other useful packages 
import importlib
from pathlib import Path
import random
from datetime import datetime

importlib.reload( uc )

"""
| Code | Meaning                |
| ---- | ---------------------- |
| `%Y` | 4-digit year           |
| `%y` | 2-digit year           |
| `%m` | month (01–12)          |
| `%b` | abbreviated month name |
| `%B` | full month name        |
| `%d` | day of month           |
| `%H` | hour (24-hr)           |
| `%M` | minute                 |
| `%S` | second                 |
| `%a` | short weekday          |
| `%A` | full weekday           |
"""

def drive(write_file=True, return_dataset=False, verbose=False ):  

    user = os.getenv("USER")  
    key = random.randint(0, 99999)
    #print(f"{key:05d}")
    sTag=f"e{key:05d}"
    now = datetime.now()
    print(f'Running the script on {now.strftime("%Y-%m-%d")}', flush=True )

    
    #####################################
    # Initialize config
    config = uc.initialize()
    print( config )
    #####################################
    # Read YAML and make date string
    #########
    file_path = './config_ana.yaml'  # Specify the path to your config file
    config = uc.read_config_yaml( file_path )
    print( config )
    year=int( config['year'] )
    month=int( config['month'] )
    day=config['day']
    hour=config['hour'] 
    #print( year, type(year).__name__ , month, type(month).__name__ )
    #print( day, type(day).__name__ , hour, type(hour).__name__ )
    print( f"Year={year},  Month={month}", flush=True )

    case=config['Case']
    if (config['Archive_base'] is None):
        archive_base = f'/glade/derecho/scratch/juliob/archive/'
    else:
        archive_base = config['Archive_base']

    day=handle(day)
    hour=handle(hour)
    #print( day, type(day).__name__ , hour, type(hour).__name__ )

    if (day==99): 
        days_to_do=np.arange( uti.days_in_month(year,month) )+1
    else: 
        days_to_do=[day]
    if (hour==99): 
        if (config['frequency'] is None):
            freq=6
        else:
            freq=config['frequency']
        hours_to_do=np.arange( start=0,stop=24, step=freq )
    else: 
        hours_to_do=[hour]

    """
    # Override YAML settings with explicit command-line arguments
    odir = args.output_dir or config.get("output_dir")
    casename = args.casename or config.get("casename")
    runtype = args.type or config.get("type")
    clean_case = args.clean_case if args.clean_case else config.get("clean_case", False)
    run_case = args.run_case if args.run_case else config.get("run_case", False)
    build_case = args.build_case if args.build_case else config.get("build_case", False)
    """
    #regrid_list=config.get("varlist") if config.get("varlist") else ['PS','U','V','OMEGA','T']
    #regrid_list=config.get("varlist") or ['PS','U','V','OMEGA','T']

    print( f'config.get("varlist") is None {config.get("varlist") is None}' )
    regrid_list=config.get("varlist") or ['PS','U','V','OMEGA','T']
    hsPat = config.get("hsPat") or 'cam.h1i'
    """
    if config.get("varlist") is not None:
        regrid_list=config.get("varlist")
    else: 
        regrid_list=['PS','U','V','OMEGA','T']
    """
    print( f"Will do days={days_to_do}", flush=True )
    print( f"Will do hours={hours_to_do}", flush=True )

    print( 'regrid_list', regrid_list , flush=True )
    print ( 'hsPat', hsPat , flush=True )

    
    #####################################
    # Initialize regrid-object library
    RgObLib={}

    RgOb_ne240_x_fv1x1  = GrU.regrid_object_lib(RgOb=RgObLib, src='ne240pg3', dst='fv1x1',    RegridMethod='CONSERVE_2ND')

    lat1R,lon1R = GrU.latlon( grid='fv1x1' )

    for d in days_to_do:
        for h in hours_to_do:
            date_tag = f"{year:04}-{month:02}-{d:02}-{3600*h:05}"
        
            #fin = f'/glade/derecho/scratch/juliob/archive/c153_topfix_ne240pg3_FMTHIST_xic_x02/atm/hist/c153_topfix_ne240pg3_FMTHIST_xic_x02.cam.h1i.{date_tag}.nc'
            fin = f"{archive_base}/{case}/atm/hist/{case}.{hsPat}.{date_tag}.nc"
            
            if Path(fin).is_file():            
                print( f"Processing {fin} ", flush=True )
                X=xr.open_dataset( fin )
    
                ##################################################################
                # Set up dataset for regridded data
                ##################################################################
                if (config["Output_abs_dir"] is None):
                    Bdiro=f"{archive_base}/{case}/atm/fv1x1"  #{case}.cam.h1i.{date_tag}.nc"
                else:
                    Bdiro=f"{config["Output_abs_dir"]}"
                #######
                os.makedirs( Bdiro , exist_ok=True )
    
                #fout = f"{Bdiro}/{case}.{sTag}.{hsPat}.{date_tag}.nc"
                fout = f"{Bdiro}/{case}.{hsPat}.{date_tag}.nc"
                
                print( f"Will write {fout}", flush=True  )
                
                coords = dict( 
                    lon  = ( ["lon"],lon1R ),
                    lat  = ( ["lat"],lat1R ),
                    lev  = ( ["lev"],X.lev.values),
                    ilev = ( ["ilev"],X.ilev.values),
                    nbnd = ( ["nbnd"], np.array( [0,1] ) ),
                    time = ( ["time"],  X.time.values ), #pd.to_datetime( pdTime_ERA[itim] ) ),
                )
    
    
                Xo = xr.Dataset( coords=coords  )
                Xo["time_bounds"] = X.time_bounds 
                Xo["hyai"] = X.hyai
                Xo["hybi"] = X.hybi
                Xo["hyam"] = X.hyam
                Xo["hybm"] = X.hybm
    
                nt,nz,ny,nx = len( X.time.values ) , len( X.lev.values ), len( lat1R), len( lon1R)
            
                ##################################################################
    
    
                lonO = X.lon.values
                latO = X.lat.values
                lev  = X.lev.values
            
                
                for var in regrid_list:

                    attribs = X[var].attrs
                    if ('long_name' in attribs):
                        longname = attribs['long_name']
                    else:
                        longname = 'n/a'
                    if ('units' in attribs):
                        units = attribs['units']
                    else:
                        units = 'n/a'

                    varO = X[var].values
                    if ('lev' in X[var].dims):
                        vdims = ('time','lev','lat','lon',)
                        reshp = [ nt,nz,ny,nx ]
                    elif ('ilev' in X[var].dims):
                        vdims = ('time','ilev','lat','lon',)
                        reshp = [ nt,nz+1,ny,nx ]
                    else: 
                        vdims = ('time','lat','lon',)
                        reshp = [ nt,ny,nx ]
                    
                    if( verbose==True):
                        print( f" regridding {var}" , flush=True )
                        print( f"    original shape {varO.shape}" , flush=True )
                        print( f"    new dims {vdims}" , flush=True )
                        print( f"    re-shape {reshp}" , flush=True )
                    
                
                    varOx1R = RgF.Horz(xfld_Src=varO ,  Src='ne240pg3' , Dst='fv1x1', RegridObj_In=  RgOb_ne240_x_fv1x1  ) 
                    
                    Dar = xr.DataArray( data=varOx1R.reshape( reshp ), 
                                        dims=vdims ,
                                        attrs=dict( long_name=longname,units=units,) ,) 
                    Xo[var] = Dar
                    if( verbose==True):
                        print( f"Finshed with {var}" , flush=True )
                    
                if (write_file==True):
                    Xo.to_netcdf( fout )
                    print( f"   Wrote {fout} ",flush=True )
            else:
                print( f"{fin} DOES NOT EXIST!!!", flush=True )
                
        
    if (return_dataset==True ):
        return Xo
    else:
        return

def procX( casename=None, year=None, month=None, day=None, sec=None, RgOb=None, momentum_fluxes=False ):


    

    for fld in X:
        print( f'{fld} {X[fld].dims}' )


def handle(var, *, hard_exit: bool = True):
    """
    Rules:
      - str "*"            -> 99
      - any str -> try int -> e.g., "7", "42", "-3" -> int value
      - plain int (not bool) -> var
      - other Integral      -> int(var)
      - otherwise           -> Bad Input -> exit(1) or raise ValueError

    hard_exit:
      True  -> sys.exit(1) on bad input
      False -> raise ValueError("Bad Input")
    """
    match var:
        case str() as s:
            if s == "*":
                return 99
            try:
                return int(s)
            except ValueError:
                if hard_exit:
                    print("Bad Input"); sys.exit(1)
                raise ValueError("Bad Input") from None

        case int() if not isinstance(var, bool):
            return var

        case numbers.Integral():
            return int(var)

        case _:
            if hard_exit:
                print("Bad Input"); sys.exit(1)
            raise ValueError("Bad Input")

if __name__ == "__main__":
    drive()


