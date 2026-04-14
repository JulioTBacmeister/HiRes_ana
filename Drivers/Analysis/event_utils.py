###!/usr/bin/env python
################################################
# New style 
# ###############################################
import sys
rootdir_ = '../'
if ( rootdir_ not in sys.path ):
    sys.path.append(rootdir_)
    print( f" a path to {rootdir_} added in {__name__} ")


from Utils import GridUtils as GrU
from Utils import MakePressures as MkP
from Utils import utils as uti
from Utils import MyConstants as Co
from Utils import time_utils as tuti
from Utils import numerical_utils as nuti

import analysis_utils as auti
import file_utils as futi


#from PyRegridding.Utils import MakePressures as MkP
#from Drivers import RegridField as RgF
import RegridField as RgF

# The usual
from datetime import date
import numpy as np
import xarray as xr
import pandas as pd

# Some other useful packages 
import copy
import time
import cftime
import yaml
import numbers

# Some other useful packages 
import importlib
from pathlib import Path


importlib.reload( auti )
importlib.reload( futi )

def make_ds(fld=None, lat=None, lon=None, zlev=None, time=None, 
            thresh=None, zlev_event=None, 
            write_ncfile=False, return_list=False, A=None ):


    
    nt,nz,ny,nx = np.shape( fld )
    
    z_event  = np.argmin( np.abs( zlev - zlev_event ) )
    print( f"Events at Z={zlev[ z_event ]}" )
    
    event_list=[]
    
    for t in np.arange( nt ):
        #events  = auti.find_gw_events(epwp= rho_epwp[t,z,:,:] , thresh=0.01, connectivity=8)
        events = auti.find_gw_events_watershed(epwp= fld[t, z_event, :, :] , thresh=thresh )  #, thresh=0.01, connectivity=8)
        event_list.append( events )
    
    if return_list == True:
        return event_list

    ####################################################################
    #                  Make DataSet
    ####################################################################
    event_id = 0
    records = []
    
    for t, events in enumerate(event_list):
        for ev in events:
            records.append({
                "event_id": event_id,
                "itime": int(t),
                "iy": int(ev["iy"]),
                "ix": int(ev["ix"]),
                "time_event": time[int(t)],
                "lat_event": lat[int(ev["iy"])],
                "lon_event": lon[int(ev["ix"])],
                "epwp_max": float(ev["epwp_max"]),
                "size": int(ev["size"]),
                "epwp_sum": float(ev["epwp_sum"]),
            })
            event_id += 1
    
    df = pd.DataFrame(records)
    
    ds = xr.Dataset.from_dataframe(df)
    
    ds = ds.assign_coords(
        lat=("lat", lat),
        lon=("lon", lon),
        zlev=("zlev", zlev),
    )
    
       

    if write_ncfile == True:
        ds.attrs["description"] = "GW events from resolved momentum flux"
        ds.attrs["epwp_definition"] = "sqrt(upwp^2 + vpwp^2)"
        ds.attrs["threshold"] = thresh
        ds.attrs["vertical_level"] = zlev[z_event]
        ds.attrs["source_files"] = f"{A.base_file_name}.%y-%m-%d-%s.nc"
        ds.attrs["start_date"] = f"{A.start_date}"
        ds.attrs["step_size_in_hours"] = f"{A.step_size}"
        outfile=f"{A.case}_Events_epwp{thresh:.0e}_Z{0.001*zlev[z_event]:.0f}km_{A.start_date}_{A.end_date}.nc"
        print( f"writing {outfile}" )
        ds.to_netcdf( outfile )

    
    return ds

def revise_ds(ds=None,fld=None,thresh=None ):

    if thresh is not None:
        if len(thresh)==1:
            thr0=thresh
        elif len(thresh)==2:
            thr0,thr1=thresh
    else:
        return -999

    

    return ds_rev

