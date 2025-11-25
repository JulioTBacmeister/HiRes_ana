#!/usr/bin/env python
################################################
# New style 
# ###############################################
import sys
import os

rootdir_ = '../'
if ( rootdir_ not in sys.path ):
    sys.path.append(rootdir_)
    print( f" a path to {rootdir_} added in {__name__} ")


from Utils import GridUtils as GrU
from Utils import utils as uti
from Utils import MyConstants as Co

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

# Constants
grav=Co.grav()


def read_in_data():

    file_path = './config_analysis_control.yaml'  # Specify the path to your config file
    config = uc.read_config_yaml( file_path )
    print( config )
    year=int( config['year'] )
    month=int( config['month'] )
    days_to_do=np.arange( uti.days_in_month(year,month) )+1
    freq=config.get('frequency') or 6
    hours_to_do=np.arange( start=0,stop=24, step=freq )

    print( f"Will do days={days_to_do}", flush=True )
    print( f"Will do hours={hours_to_do}", flush=True )

    case=config['Case']
    if (config['Archive_base'] is None):
        archive_base = f'/glade/derecho/scratch/juliob/archive/'
    else:
        archive_base = config['Archive_base']

    slices=[]
    for d in days_to_do:
        for h in hours_to_do:
            date_tag = f"{year:04}-{month:02}-{d:02}-{3600*h:05}"
            if (config["Output_abs_dir"] is None):
                Bdiro=f"{archive_base}/{case}/atm/fv1x1"  #{case}.cam.h1i.{date_tag}.nc"
            else:
                Bdiro=f"{config["Output_abs_dir"]}"
            #######
            os.makedirs( Bdiro , exist_ok=True )

            fin = f"{Bdiro}/{case}.cam.h1i.{date_tag}.nc"

            if Path(fin).is_file():            
                print( f"Processing {fin} ", flush=True )
                X=xr.open_dataset( fin )
                rupwp = X.upwp.values[0,20,:,:]
                slices.append(rupwp)

    ayx = np.stack(slices, axis=0)        # shape = (nt, ny, nx)

    return ayx

                
