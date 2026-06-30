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
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

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

# This allows for both dict.key and dict['key'] syntax
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


###############################################################################
def make_El(A=None, 
            fractions_for_thresholds=None, 
            zlev_event=None, 
            lat_range=None,lon_range=None,exclude_orography=True,
            peak_footprint=(3,3),
            return_after_stage1=False):

    event_fld = 'rho_epwp'
    if fractions_for_thresholds is not None:
        fracs = fractions_for_thresholds
    else:
        fracs = [0.995,0.90,0.50,0.25,0.125,0.0625]

    
    time, zlev, lat, lon = A.time, A.zlev, A.lat, A.lon

    topo_thresh=1.e-12 # needs to be this low to remove Malvinas/Falklands(?)  ... 0.0001 leaves them. 

    ###############################################
    # Stage 1.
    # Scope out range and distribution of events
    ###############################################
    z_event  = np.argmin( np.abs( zlev - zlev_event ) )
    x=A.rho_epwp[:,z_event,:,:].flatten()
    cumu,xs=auti.cumul_big_to_small( x, plot_it=True )
    
    thresh_fracs=[]
    thresholds=[]
    N_events=[]
    Total_events=len(xs)
    print( f"looking at Z={zlev[ z_event ]}" )
    print( f"There are a total of {Total_events} gridpoints in {lon_range}X{lat_range} " )
    for b in fracs:
        xoo=np.argmin( np.abs(cumu-b) )
        print(f" fraction={100*b:5.2f}% of total epwp is in events with epwp > {xs[xoo]:.5f}. Carried by N={len(xs[xoo:]):6d} events or {100*len(xs[xoo:])/len(xs):5.2f}%  ")
        thresh_fracs.append(xs[xoo])
        thresholds.append( np.array([ xs[xoo],1.e6] ) )
        N_events.append( len(xs[xoo:]) )

    print(f" Threshholds {thresholds}")
    
    thresh=[0.,1e6]
    second_thresh=None #second_thresholds[ithr]
    ds =  make_ds(fld=A.rho_epwp[:,:,:,:], lon=lon, lat=lat, zlev=zlev, time=time, 
                     thresh=thresh,second_thresh=second_thresh,zlev_event=zlev_event, 
                     lat_range=lat_range, lon_range=lon_range,
                     peak_footprint=peak_footprint )
    
    # get shape of varaiables
    nt,nz,ny,nx = np.shape( A.u )
    
    htopo_t = np.tile(A.htopo[None, :, :], ( nt, 1, 1))
    htopo_4D_x , time4D,lat4D,lon4D  = auti.cube4D_ds( event_ds=ds, aa=htopo_t , lon=lon, lat=lat, window=[0,5,5] , TZHkey='tyx', lat_range=[-999,999], lon_range=[-999,999] )
    
    htopo_MMM = auti.collapseSpace( htopo_4D_x , TZHkey='etyx')
    htopo_super_max = htopo_MMM[2].max( axis=1 )

    print( f"N events: = {ds.sizes['index']} BEFORE any topo filtering !!!! " )
 
    ########################################
    # Exclude events with topography nearby
    ########################################
    if (exclude_orography==True):
        flat=np.where(htopo_super_max<topo_thresh) #1.e-12  ) #0.0001)
        flat[0].shape
        ds_flat=ds.isel( index=flat[0] )    
        ds=ds_flat
        print( f"N events: = {ds.sizes['index']} AFTER topo filtering !!!! " )
    
    
    x=ds.epwp_max.values
    cumu,xs=auti.cumul_big_to_small( x, plot_it=True )
    
    thresh_fracs=[]
    thresholds=[]
    N_events=[]
    Total_events=len(xs)
    print( f"There are a total of {Total_events} events in {lon_range}X{lat_range}, exclude orography={exclude_orography} " )
    for b in fracs:
        xoo=np.argmin( np.abs(cumu-b) )
        print(f" fraction={100*b:5.2f}% of total epwp is in events with epwp > {xs[xoo]:.5f}. Carried by N={len(xs[xoo:]):6d} events or {100*len(xs[xoo:])/len(xs):5.2f}%  ")
        thresh_fracs.append(xs[xoo])
        thresholds.append( np.array([ xs[xoo],1.e6] ) )
        N_events.append( len(xs[xoo:]) )

    print(f" Threshholds {thresholds}")


    if return_after_stage1==True:
        return ds
    
    ############################################################################
    #   STAGE 2. Actually construct El objects using statistics from stage 1
    ############################################################################
    
    El=[]
    
    print( f"This run use dycore={A.dycore}")
    
    if A.dycore == 'MPAS':
        ## for MPAS 3km
        big_window=[3,5,5]
        lil_window=[3,2,2]
        
    elif A.dycore == 'SE':
        # for ne240
        big_window=[2,5,5]
        lil_window=[2,2,2]
    
    
    ithr=0
    for thresh in thresholds:
        second_thresh=None #second_thresholds[ithr]
        ds =   make_ds(fld=A.rho_epwp[:,:,:,:], lon=lon, lat=lat, zlev=zlev, time=time, 
                         thresh=thresh,second_thresh=second_thresh,zlev_event=zlev_event, 
                         lat_range=lat_range, lon_range=lon_range,
                         peak_footprint=peak_footprint )
        
    
        # get shape of varaiables
        nt,nz,ny,nx = np.shape( A.u )
        
        htopo_t = np.tile(A.htopo[None, :, :], ( nt, 1, 1))
        htopo_4D_x , time4D,lat4D,lon4D  = auti.cube4D_ds( event_ds=ds, aa=htopo_t , lon=lon, lat=lat, window=[0,5,5] , TZHkey='tyx', lat_range=[-999,999], lon_range=[-999,999] )
        
        htopo_MMM = auti.collapseSpace( htopo_4D_x , TZHkey='etyx')
        htopo_super_max = htopo_MMM[2].max( axis=1 )
        print( f"N events: = {ds.sizes['index']} before topo filtering !!!! " )
        
        ########################################
        # Exclude events with topography nearby
        ########################################
        if (exclude_orography==True):
            flat=np.where(htopo_super_max<topo_thresh) #0.0001)
            flat[0].shape
            ds_flat=ds.isel( index=flat[0] )    
            ds=ds_flat
            print( f"N events: = {ds.sizes['index']} AFTER topo filtering !!!! " )

                    
        E_ = {'ds':ds }
        E_['fld'] = event_fld
        E_['case'] = A.case
        E_['dycore'] = A.dycore
        E_['start_date'] = A.start_date
        E_['end_date'] = A.end_date
        E_['threshold']=thresh
        E_['zlev_event']=zlev_event
        if second_thresh is not None:
            E_["second_threshold"] = second_thresh        
        E_['exclude_orography'] = exclude_orography
        E_['lat_range']=lat_range
        E_['lon_range']=lon_range
        E_['N_events'] = ds.sizes['index']
        E_['Total_events'] = Total_events # after Topo filtering but BEFOR any thresholds applied
        E_['Frac_of_total_epwp'] = fracs[ithr]

        # Wrap in coordinate vectors
        # time, zlev, lat, lon = A.time, A.zlev, A.lat, A.lon
        E_['timeA'], E_['zlevA'], E_['latA'], E_['lonA'] = time,zlev,lat,lon

        
        window=lil_window # [3,2,2]
        #window=[6,2,2] # MPAS results are 3-hourly ...
        if ds.sizes['index'] < 75_000:
            window = big_window #[3,5,5] #[6,5,5]
            print( f"Big window ")
        #window=[6,5,5] # MPAS results are 3-hourly ...
        
        
        E_['window_tyx']=window
        E_['peak_footprint']=peak_footprint
        
        htopo_4D , time4D,lat4D,lon4D  = auti.cube4D_ds( event_ds=ds, aa=htopo_t , lon=lon, lat=lat, window=window , TZHkey='tyx', lat_range=lat_range, lon_range=lon_range )
        zeta_4D, time4D,lat4D,lon4D = auti.cube4D_ds( event_ds=ds, aa=A.zeta , lon=lon, lat=lat, window=window , TZHkey='tzyx', lat_range=lat_range, lon_range=lon_range )
        tilt_4D, time4D,lat4D,lon4D = auti.cube4D_ds( event_ds=ds, aa=A.tilt , lon=lon, lat=lat, window=window , TZHkey='tzyx', lat_range=lat_range, lon_range=lon_range )
        fgf_4D, time4D,lat4D,lon4D  = auti.cube4D_ds( event_ds=ds, aa=A.fgf , lon=lon, lat=lat, window=window , TZHkey='tzyx', lat_range=lat_range, lon_range=lon_range )
        epwp_4D, time4D,lat4D,lon4D = auti.cube4D_ds( event_ds=ds, aa=A.rho_epwp , lon=lon, lat=lat, window=window , TZHkey='tzyx', lat_range=lat_range, lon_range=lon_range )
        upwp_4D, time4D,lat4D,lon4D = auti.cube4D_ds( event_ds=ds, aa=A.rho_upwp , lon=lon, lat=lat, window=window , TZHkey='tzyx', lat_range=lat_range, lon_range=lon_range )
        vpwp_4D, time4D,lat4D,lon4D = auti.cube4D_ds( event_ds=ds, aa=A.rho_vpwp , lon=lon, lat=lat, window=window , TZHkey='tzyx', lat_range=lat_range, lon_range=lon_range )
        u_4D, time4D,lat4D,lon4D    = auti.cube4D_ds( event_ds=ds, aa=A.u , lon=lon, lat=lat, window=window , TZHkey='tzyx', lat_range=lat_range, lon_range=lon_range )
        v_4D, time4D,lat4D,lon4D    = auti.cube4D_ds( event_ds=ds, aa=A.v , lon=lon, lat=lat, window=window , TZHkey='tzyx', lat_range=lat_range, lon_range=lon_range )
        th_4D, time4D,lat4D,lon4D   = auti.cube4D_ds( event_ds=ds, aa=A.th , lon=lon, lat=lat, window=window , TZHkey='tzyx', lat_range=lat_range, lon_range=lon_range )
        stab_4D, time4D,lat4D,lon4D = auti.cube4D_ds( event_ds=ds, aa=A.stab , lon=lon, lat=lat, window=window , TZHkey='tzyx', lat_range=lat_range, lon_range=lon_range )
    
        E_['time4D'], E_['lat4D'], E_['lon4D'] = time4D,lat4D,lon4D
        E_['u_4D'], E_['v_4D'], E_['htopo_4D'] = u_4D,v_4D,htopo_4D
        E_['zeta_4D'], E_['tilt_4D'] , E_['fgf_4D']  = zeta_4D,tilt_4D,fgf_4D
        E_['upwp_4D'], E_['vpwp_4D'] , E_['epwp_4D']  = upwp_4D,vpwp_4D,epwp_4D
        E_['th_4D'] = th_4D
        E_['stab_4D'] = stab_4D
        
        if 'precl' in A:
            precl_4D , time4D,lat4D,lon4D  = auti.cube4D_ds( event_ds=ds, aa=A.precl , lon=lon, lat=lat, window=window , TZHkey='tzyx', lat_range=lat_range, lon_range=lon_range )
            E_['precl_4D'] = precl_4D
        if 'rho_thpwp' in A:
            thpwp_4D , time4D,lat4D,lon4D  = auti.cube4D_ds( event_ds=ds, aa=A.rho_thpwp , lon=lon, lat=lat, window=window , TZHkey='tzyx', lat_range=lat_range, lon_range=lon_range )
            E_['thpwp_4D'] = thpwp_4D
    
        E = AttrDict( E_ )
        El.append(E)
        print( f" fininshed threshold = {thresh}. N events: = {ds.sizes['index']} " )
        ithr=ithr+1

    return El


###############################################################################
def make_ds(fld=None, lat=None, lon=None, zlev=None, time=None, 
            thresh=None, second_thresh=None, zlev_event=None, 
            lat_range=None,lon_range=None, peak_footprint=(3,3),
            write_ncfile=False, return_list=False, A=None ):


    
    nt,nz,ny,nx = np.shape( fld )
    
    z_event  = np.argmin( np.abs( zlev - zlev_event ) )
    print( f"Events at Z={zlev[ z_event ]}" )
    print( f"Peak footprint={peak_footprint[0]}X{peak_footprint[1]}" )
    
    event_list=[]
    total_events = 0
    for t in np.arange( nt ):
        #events  = auti.find_gw_events(epwp= rho_epwp[t,z,:,:] , thresh=0.01, connectivity=8)
        events = auti.find_gw_events_watershed_2(epwp= fld[t, z_event, :, :], thresh=thresh, second_thresh=second_thresh, 
                                                lat_range=lat_range, lon_range=lon_range, lon=lon, lat=lat, 
                                                peak_footprint=  peak_footprint )
        event_list.append( events )
        #print( f"Length of event list in time loop {len(events)}" )
        total_events = total_events + len(events)
    
    print( f"Total events {total_events} in {lon_range}X{lat_range} for thresh={thresh} BEFORE topo filtering. Peak footprint={peak_footprint[0]}X{peak_footprint[1]}" )
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
    
       
    
    return ds
    
######################################################################################
def revise_ds(ds=None,fld=None,thresh=None ):

    if thresh is not None:
        if len(thresh)==1:
            thr0=thresh
        elif len(thresh)==2:
            thr0,thr1=thresh
    else:
        return -999

    

    return ds_rev

#######################################################################
def ds_to_event_list(ds, nt=None):
    """
    Convert xarray event dataset back to list of lists format.
    
    Parameters
    ----------
    ds : xr.Dataset
        As produced by make_ds(), with variables:
        event_id, itime, iy, ix, epwp_max, size, epwp_sum,
        lat_event, lon_event, time_event
    nt : int or None
        Total number of timesteps. If None, inferred from ds.itime.max()+1.
        Provide explicitly if the last timestep(s) had no events.
    
    Returns
    -------
    event_list : list of lists
        event_list[t] is a list of event dicts at timestep t,
        each with keys matching the original find_gw_events_watershed_2 output
        plus 't' (timestep index) for tracking.
    """
    if nt is None:
        nt = int(ds.itime.values.max()) + 1

    # initialise empty list for each timestep
    event_list = [[] for _ in range(nt)]

    # convert dataset to pandas for easy iteration
    df = ds.to_dataframe().reset_index()

    for _, row in df.iterrows():
        t = int(row['itime'])
        event_list[t].append({
            'iy':       int(row['iy']),
            'ix':       int(row['ix']),
            'epwp_max': float(row['epwp_max']),
            'size':     int(row['size']),
            'epwp_sum': float(row['epwp_sum']),
            't':        t,    # needed for tracking
        })

    # report
    total = sum(len(evs) for evs in event_list)
    empty = sum(1 for evs in event_list if len(evs) == 0)
    print(f"Reconstructed {total} events across {nt} timesteps")
    print(f"  {empty} timesteps had no events")

    return event_list
#######################################################################
def combine_event_dicts(El_strong, El_weak, label_key='event_strength'):
    """
    Combine two event dicts of the same structure into one,
    adding a source label array to track provenance.
    
    Parameters
    ----------
    El_strong : dict
        Event dict for strong events.
    El_weak : dict
        Event dict for weak events.
    label_key : str
        Key name for the source label array added to the combined dict.
    
    Returns
    -------
    El_combined : dict
        Combined dict with all fields concatenated along the event axis (axis=0).
    source_labels : np.ndarray, shape (n_events_total,)
        1 for strong events, 0 for weak events. Useful for validation later.
    split_idx : int
        Index where weak events begin in the combined array.
        i.e. El_combined[key][:split_idx] are strong, [split_idx:] are weak.
    """
    keys = list(El_strong.keys())
    
    # sanity check both dicts have the same keys
    assert set(keys) == set(El_weak.keys()), \
        "Dicts have different keys — check your inputs"
    
    El_combined = {}
    skip_keys = []  # keys we can't concatenate (e.g. scalars, metadata)

    for key in keys:
        val_s = El_strong[key]
        val_w = El_weak[key]

        # attempt concatenation along event axis
        try:
            El_combined[key] = np.concatenate([val_s, val_w], axis=0)
        except (ValueError, TypeError):
            # non-array fields (e.g. ds, scalar metadata) — Delete/keep strong version
            #El_combined[key] = val_s
            skip_keys.append(key)

    split_idx = El_strong[keys[-1]].shape[0]  # where weak events begin

    # source label array: 1=strong, 0=weak
    n_strong = split_idx
    n_weak   = El_weak[keys[-1]].shape[0]
    source_labels = np.array([1]*n_strong + [0]*n_weak)

    El_combined[label_key] = source_labels

    if skip_keys:
        print(f"Note: these keys were not concatenated (kept strong version): {skip_keys}")

    print(f"Combined: {n_strong} strong + {n_weak} weak = {n_strong+n_weak} total events")

    Eco = AttrDict( El_combined )
    #return El_combined, source_labels, split_idx
    return Eco


def subsample_event_dicts(El, indices):
    """
    Subsample event dict on proviede indices of 
    
    Parameters
    ----------
    El : dict
        Event dict for events.
    indices : int
        indices on which to subsample
    
    Returns
    -------
    El_x : subsampled dict
    """
    keys = list(El.keys())
    El_x_ = {}
    for key in keys:
        val = El[key]
        try:
            El_x_[key] = val[indices]
        except (ValueError, TypeError):
            print(f" Couldn't subsmaple f{key}")



    print(f"Subsampled: {key} to {len(indices)} events")

    El_x = AttrDict( El_x_ )
    return El_x
    
def avg_over_v(El,verbose=False):
    """
    Subsample event dict on proviede indices of 
    
    Parameters
    ----------
    El : dict
        Event dict for events.
    indices : int
        indices on which to subsample
    
    Returns
    -------
    El_x : subsampled dict
    """
    keys = list(El.keys())
    El_x_ = {}
    for key in keys:
        val = El[key]
        try:
            El_x_[key] = val.mean(axis=0)
        except (ValueError, TypeError, AttributeError):
            if verbose == True:
                print(f" Couldn't average {key} over events")

    El_x = AttrDict( El_x_ )
    return El_x



def spec_4D_cube_ranges( El=None, trange=None, yrange=None, xrange=None ):

    time,lat,lon=El.timeA,El.latA,El.lonA
    window_tyx=El.window_tyx



###############################################################################
#    Tracking 
###############################################################################

###############################################################################
#    Tracking 
###############################################################################

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute haversine distance in km between two points.
    """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def haversine_matrix(lats1, lons1, lats2, lons2):
    """
    Compute pairwise haversine distances in km between two sets of points.
    Returns matrix of shape (len(lats1), len(lats2)).
    """
    R = 6371.0
    lats1_r = np.radians(lats1)[:, None]
    lons1_r = np.radians(lons1)[:, None]
    lats2_r = np.radians(lats2)[None, :]
    lons2_r = np.radians(lons2)[None, :]

    dlat = lats2_r - lats1_r
    dlon = lons2_r - lons1_r
    a    = (np.sin(dlat/2)**2 +
            np.cos(lats1_r) * np.cos(lats2_r) * np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))


def track_events(event_list, lat, lon, dt_hours=3.0,
                 max_speed_kmh=80.0, min_lifetime=2):
    """
    Track GW events across timesteps.

    Parameters
    ----------
    event_list : list of lists
        event_list[t] is a list of event dicts at timestep t,
        each with keys 'iy', 'ix', 'epwp_max', 't'.
    lat : np.ndarray, shape (ny,)
    lon : np.ndarray, shape (nx,)
    dt_hours : float
        Timestep interval in hours.
    max_speed_kmh : float
        Maximum physically reasonable storm speed in km/h.
        Events further apart than max_speed_kmh * dt_hours are not linked.
    min_lifetime : int
        Minimum number of timesteps for a track to be retained.
        Filters out single-timestep noise events.

    Returns
    -------
    tracks : list of dicts
        Each track dict contains:
        - 'track_id'   : int
        - 'times'      : list of int timestep indices
        - 'iy'         : list of int y indices
        - 'ix'         : list of int x indices
        - 'lats'       : list of float latitudes
        - 'lons'       : list of float longitudes
        - 'epwp_max'   : list of float peak epwp at each timestep
        - 'lifetime'   : int number of timesteps
        - 'vel_y_kmh'  : np.ndarray velocity in y direction (km/h)
        - 'vel_x_kmh'  : np.ndarray velocity in x direction (km/h)
        - 'speed_kmh'  : np.ndarray total speed (km/h)
        - 'mean_speed' : float mean speed over track lifetime
    """
    max_dist_km = max_speed_kmh * dt_hours   # max displacement per timestep

    # --- initialise tracks from first non-empty timestep ------------------
    track_id_counter = 0
    active_tracks    = []   # tracks still being extended

    # find first timestep with events
    t_start = next(t for t, evs in enumerate(event_list) if len(evs) > 0)

    for ev in event_list[t_start]:
        active_tracks.append({
            'track_id': track_id_counter,
            'times':    [ev['t']],
            'iy':       [ev['iy']],
            'ix':       [ev['ix']],
            'lats':     [lat[ev['iy']]],
            'lons':     [lon[ev['ix']]],
            'epwp_max': [ev['epwp_max']],
        })
        track_id_counter += 1

    completed_tracks = []

    # --- step through time ------------------------------------------------
    for t in range(t_start + 1, len(event_list)):
        current_events = event_list[t]

        if len(current_events) == 0:
            # no events this timestep — terminate all active tracks
            completed_tracks.extend(active_tracks)
            active_tracks = []
            continue

        if len(active_tracks) == 0:
            # no active tracks — start new ones from current events
            for ev in current_events:
                active_tracks.append({
                    'track_id': track_id_counter,
                    'times':    [ev['t']],
                    'iy':       [ev['iy']],
                    'ix':       [ev['ix']],
                    'lats':     [lat[ev['iy']]],
                    'lons':     [lon[ev['ix']]],
                    'epwp_max': [ev['epwp_max']],
                })
                track_id_counter += 1
            continue

        # --- compute distances between active track last positions
        #     and current events ------------------------------------------
        track_lats = np.array([tr['lats'][-1] for tr in active_tracks])
        track_lons = np.array([tr['lons'][-1] for tr in active_tracks])
        event_lats = np.array([lat[ev['iy']] for ev in current_events])
        event_lons = np.array([lon[ev['ix']] for ev in current_events])
        event_amps = np.array([ev['epwp_max'] for ev in current_events])

        dist = haversine_matrix(track_lats, track_lons,
                                event_lats, event_lons)
        # shape: (n_active_tracks, n_current_events)

        # mask out links that exceed max displacement
        dist[dist > max_dist_km] = np.inf

        # --- greedy matching: assign each event to nearest active track ---
        # process events in order of decreasing amplitude (strong events
        # get first pick of track assignments)
        event_order      = np.argsort(event_amps)[::-1]
        track_matched    = np.zeros(len(active_tracks), dtype=bool)
        event_matched    = np.zeros(len(current_events), dtype=bool)
        track_assignment = {}   # event_idx -> track_idx

        for ei in event_order:
            # find nearest unmatched track within max_dist
            dists_to_tracks = dist[:, ei].copy()
            dists_to_tracks[track_matched] = np.inf   # exclude already matched

            nearest_track = np.argmin(dists_to_tracks)
            if dists_to_tracks[nearest_track] < np.inf:
                track_assignment[ei]         = nearest_track
                track_matched[nearest_track] = True
                event_matched[ei]            = True

        # --- handle mergers -----------------------------------------------
        # if multiple events map to same track, keep largest (already handled
        # by greedy amplitude-sorted matching above — first event wins)

        # --- update matched tracks ----------------------------------------
        new_active_tracks = []
        for ei, ti in track_assignment.items():
            ev = current_events[ei]
            active_tracks[ti]['times'].append(ev['t'])
            active_tracks[ti]['iy'].append(ev['iy'])
            active_tracks[ti]['ix'].append(ev['ix'])
            active_tracks[ti]['lats'].append(lat[ev['iy']])
            active_tracks[ti]['lons'].append(lon[ev['ix']])
            active_tracks[ti]['epwp_max'].append(ev['epwp_max'])
            new_active_tracks.append(active_tracks[ti])

        # --- terminate unmatched tracks ------------------------------------
        for ti, tr in enumerate(active_tracks):
            if not track_matched[ti]:
                completed_tracks.append(tr)

        # --- start new tracks for unmatched events (splits/new systems) ---
        for ei, ev in enumerate(current_events):
            if not event_matched[ei]:
                new_active_tracks.append({
                    'track_id': track_id_counter,
                    'times':    [ev['t']],
                    'iy':       [ev['iy']],
                    'ix':       [ev['ix']],
                    'lats':     [lat[ev['iy']]],
                    'lons':     [lon[ev['ix']]],
                    'epwp_max': [ev['epwp_max']],
                })
                track_id_counter += 1

        active_tracks = new_active_tracks

    # terminate any remaining active tracks
    completed_tracks.extend(active_tracks)

    # --- compute velocities and filter by lifetime ------------------------
    final_tracks = []
    for tr in completed_tracks:
        lifetime = len(tr['times'])
        if lifetime < min_lifetime:
            continue

        lats = np.array(tr['lats'])
        lons = np.array(tr['lons'])

        # velocity at each step (km/h)
        vel_y = np.zeros(lifetime)
        vel_x = np.zeros(lifetime)

        for i in range(1, lifetime):
            # meridional displacement
            dlat_km = haversine_distance(
                lats[i-1], lons[i-1], lats[i], lons[i-1]
            ) * np.sign(lats[i] - lats[i-1])
            # zonal displacement
            dlon_km = haversine_distance(
                lats[i], lons[i-1], lats[i], lons[i]
            ) * np.sign(lons[i] - lons[i-1])

            vel_y[i] = dlat_km / dt_hours
            vel_x[i] = dlon_km / dt_hours

        # forward difference for first point
        vel_y[0] = vel_y[1] if lifetime > 1 else 0
        vel_x[0] = vel_x[1] if lifetime > 1 else 0

        speed = np.sqrt(vel_y**2 + vel_x**2)

        tr['lifetime']   = lifetime
        tr['vel_y_kmh']  = vel_y
        tr['vel_x_kmh']  = vel_x
        tr['speed_kmh']  = speed
        tr['mean_speed'] = speed[1:].mean() if lifetime > 1 else 0.0

        final_tracks.append(tr)

    print(f"Tracked {len(final_tracks)} systems "
          f"(min lifetime={min_lifetime} timesteps = "
          f"{min_lifetime * dt_hours:.0f} hours)")
    print(f"  Lifetime distribution:")
    lifetimes = np.array([tr['lifetime'] for tr in final_tracks])
    for lt in [2, 4, 8, 16]:
        print(f"    >= {lt} timesteps ({lt*dt_hours:.0f}h): "
              f"{(lifetimes >= lt).sum()} tracks")
    print(f"  Mean speed: {np.mean([tr['mean_speed'] for tr in final_tracks]):.1f} km/h")

    return final_tracks

def plot_wind_rose_histo(vel_x_kmh, vel_y_kmh,
                   n_sectors=12, speed_bin_ms=5.0,
                   ax=None, title='Track velocity wind rose'):
    """
    Wind rose where:
      - all petals reach the same maximum radius (overall max speed)
      - each petal is divided into speed segments
      - each segment is shaded by its count / total events (global fraction)

    Parameters
    ----------
    vel_x_kmh : np.ndarray — zonal velocity components (km/h)
    vel_y_kmh : np.ndarray — meridional velocity components (km/h)
    n_sectors : int — number of directional sectors (12 → 30° each)
    speed_bin_ms : float — width of each speed bin in m/s
    ax : matplotlib polar Axes or None
    title : str
    """
    # convert to m/s
    vel_x = vel_x_kmh / 3.6
    vel_y = vel_y_kmh / 3.6
    speed = np.sqrt(vel_x**2 + vel_y**2)

    # meteorological direction: FROM
    math_angle = np.degrees(np.arctan2(vel_y, vel_x))
    direction  = (90 - math_angle) % 360

    # directional bins
    sector_width       = 360.0 / n_sectors
    sector_edges       = np.arange(-sector_width/2, 360, sector_width)
    sector_centres_rad = np.radians(np.arange(0, 360, sector_width))

    # speed bins
    max_speed    = np.ceil(speed.max() / speed_bin_ms) * speed_bin_ms
    speed_edges  = np.arange(0, max_speed + speed_bin_ms, speed_bin_ms)
    n_speed_bins = len(speed_edges) - 1
    speed_labels = [f'{speed_edges[i]:.0f}–{speed_edges[i+1]:.0f} m/s'
                    for i in range(n_speed_bins)]

    # counts per (sector, speed_bin) as fraction of total events
    n_total = len(speed)
    fractions = np.zeros((n_sectors, n_speed_bins))

    for s in range(n_sectors):
        d_lo = sector_edges[s]
        d_hi = sector_edges[s + 1]
        if s == 0:
            in_sector = (direction >= (360 + d_lo)) | (direction < d_hi)
        else:
            in_sector = (direction >= d_lo) & (direction < d_hi)
        for b in range(n_speed_bins):
            in_bin = (speed >= speed_edges[b]) & (speed < speed_edges[b + 1])
            fractions[s, b] = (in_sector & in_bin).sum() / n_total

    # global max fraction — for normalising colormap
    frac_max = fractions.max()

    # ── plot ────────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7),
                               subplot_kw={'projection': 'polar'})
    else:
        fig = ax.figure

    #print( 'ax for rose histo' )
    #print(ax.type)
                        
    bar_width = np.radians(sector_width) * 0.9
    cmap      = plt.cm.Blues
    norm      = plt.Normalize(vmin=0, vmax=frac_max)

    """
    for b in range(n_speed_bins):
        bottoms = pct[:, :b].sum(axis=1)
        ax.bar(sector_centres_rad, pct[:, b],
               width=bar_width, bottom=bottoms,
               color=colors[b], label=speed_labels[b],
               edgecolor='white', linewidth=0.4, alpha=0.9)
    """
    for s in range(n_sectors):
        bottom = 0.0
        for b in range(n_speed_bins):
            height = speed_edges[b + 1] - speed_edges[b]   # = speed_bin_ms
            color  = cmap(norm(fractions[s, b]))
            if s==0:
                ax.bar(sector_centres_rad[s], height,
                       width=bar_width, bottom=bottom,label=speed_labels[b],
                       color=color, edgecolor='white', linewidth=0.3)
            else:
                ax.bar(sector_centres_rad[s], height,
                       width=bar_width, bottom=bottom,
                       color=color, edgecolor='white', linewidth=0.3)
                
            bottom += height

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.75, -0.02, 0.20, 0.03])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Fraction of total events', fontsize=8)
    """
    # colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.12, shrink=0.85, aspect=20)
    cbar.set_label('Fraction of total events', fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.15, 0.02, 0.70, 0.03])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')

    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%')
    )
    
    ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.0),
              fontsize=7, title='Speed', title_fontsize=8)

              

    """
    # radial axis — speed scale, labels at 45°
    ax.set_rlabel_position(245)
    rticks = np.arange(0, max_speed + speed_bin_ms, speed_bin_ms)
    ax.set_yticks(rticks)
    ax.set_yticklabels([f'{r:.0f} m/s' for r in rticks], fontsize=7)
    ax.set_ylim(0, max_speed)

    # meteorological axes: N up, clockwise
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(np.arange(0, 360, 30)))
    ax.set_xticklabels(['N','30','60','E','120','150',
                         'S','210','240','W','300','330'], fontsize=8)

    ax.set_title(title, fontsize=10, pad=15)

    return ax
def plot_wind_rose(vel_x_kmh, vel_y_kmh,
                   n_sectors=12, speed_bin_ms=5.0,
                   ax=None, title='Track velocity wind rose'):
    """
    Wind rose where:
      - petal length = mean speed in that sector (m/s)
      - petal shading = frequency of observations in that sector (%)

    Parameters
    ----------
    vel_x_kmh : np.ndarray — zonal velocity components (km/h)
    vel_y_kmh : np.ndarray — meridional velocity components (km/h)
    n_sectors : int — number of directional sectors (12 → 30° each)
    speed_bin_ms : float — kept for API compatibility; not used for binning
    ax : matplotlib polar Axes or None
    title : str
    """
    # convert to m/s
    vel_x = vel_x_kmh / 3.6
    vel_y = vel_y_kmh / 3.6
    speed = np.sqrt(vel_x**2 + vel_y**2)

    # meteorological direction: FROM
    math_angle = np.degrees(np.arctan2(vel_y, vel_x))
    direction  = (90 - math_angle) % 360  # GW convention: TOWARDS

    # directional bins
    sector_width   = 360.0 / n_sectors
    sector_edges   = np.arange(-sector_width/2, 360, sector_width)
    sector_centres_rad = np.radians(np.arange(0, 360, sector_width))

    # per-sector statistics
    max_speed  = np.zeros(n_sectors)
    mean_speed = np.zeros(n_sectors)
    frequency  = np.zeros(n_sectors)

    for s in range(n_sectors):
        d_lo = sector_edges[s]
        d_hi = sector_edges[s + 1]
        if s == 0:
            in_sector = (direction >= (360 + d_lo)) | (direction < d_hi)
        else:
            in_sector = (direction >= d_lo) & (direction < d_hi)

        n_in = in_sector.sum()
        frequency[s]  = n_in / len(speed) * 100.0
        mean_speed[s] = speed[in_sector].mean() if n_in > 0 else 0.0
        max_speed[s] = speed[in_sector].max() if n_in > 0 else 0.0

    # ── plot ────────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw={'projection': 'polar'})
    else:
        fig = ax.figure

    bar_width = np.radians(sector_width) * 0.9

    # colour map: frequency → shading
    norm   = plt.Normalize(vmin=0, vmax=frequency.max())
    cmap   = plt.cm.Blues
    colors = cmap(norm(frequency))

    bars = ax.bar(sector_centres_rad, max_speed,
                  width=bar_width,
                  color=colors,
                  edgecolor='white', linewidth=0.5)

    # colourbar for frequency
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.12, shrink=0.7, aspect=20)
    cbar.set_label('Frequency (%)', fontsize=8)

    # radial axis = mean speed
    ax.set_ylabel('Mean speed (m/s)', labelpad=30, fontsize=8)

    # meteorological axes: N up, clockwise
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(np.arange(0, 360, 30)))
    ax.set_xticklabels(['N','30','60','E','120','150',
                         'S','210','240','W','300','330'], fontsize=8)

    ax.set_title(title, fontsize=10, pad=15)

    return ax


def plot_wind_rose_trad(vel_x_kmh, vel_y_kmh,
                   n_sectors=12, speed_bin_ms=5.0,
                   ax=None, title='Track velocity wind rose'):
    """
    Plot a meteorological wind rose from velocity components.

    Meteorological convention: direction = where the wind is coming FROM.
    North is up, angles run clockwise.

    Parameters
    ----------
    vel_x_kmh : np.ndarray — zonal velocity components (km/h)
    vel_y_kmh : np.ndarray — meridional velocity components (km/h)
    n_sectors : int — number of directional sectors (12 → 30° each)
    speed_bin_ms : float — width of each speed ring in m/s
    ax : matplotlib polar Axes or None
    title : str
    """
    # convert to m/s
    vel_x = vel_x_kmh / 3.6
    vel_y = vel_y_kmh / 3.6
    speed = np.sqrt(vel_x**2 + vel_y**2)

    # meteorological direction: FROM (so flip components)
    # math angle (from east, anticlockwise) → met direction (from north, clockwise)
    # direction = (270 - math_deg) % 360
    math_angle = np.degrees(np.arctan2(vel_y, vel_x))
    direction  = (90 - math_angle) % 360       # GW convention: TOWARDS

    # directional bins — centred so north (0°) is centred on first bin
    sector_width = 360.0 / n_sectors
    sector_edges = np.arange(-sector_width/2, 360, sector_width)
    sector_centres_rad = np.radians(
        np.arange(0, 360, sector_width)          # 0=N, 90=E, 180=S, 270=W
    )

    # speed bins
    max_speed    = np.ceil(speed.max() / speed_bin_ms) * speed_bin_ms
    speed_edges  = np.arange(0, max_speed + speed_bin_ms, speed_bin_ms)
    n_speed_bins = len(speed_edges) - 1
    speed_labels = [f'{speed_edges[i]:.0f}–{speed_edges[i+1]:.0f} m/s'
                    for i in range(n_speed_bins)]

    # colour map — light to dark blue
    cmap   = plt.cm.Blues
    colors = [cmap(0.2 + 0.8 * i / (n_speed_bins - 1))
              for i in range(n_speed_bins)]

    # bin counts: shape (n_sectors, n_speed_bins)
    counts = np.zeros((n_sectors, n_speed_bins))
    for s in range(n_sectors):
        d_lo = sector_edges[s]
        d_hi = sector_edges[s + 1]
        if s == 0:
            in_sector = (direction >= (360 + d_lo)) | (direction < d_hi)
        else:
            in_sector = (direction >= d_lo) & (direction < d_hi)
        for b in range(n_speed_bins):
            counts[s, b] = np.sum(
                in_sector &
                (speed >= speed_edges[b]) &
                (speed <  speed_edges[b + 1])
            )

    # express as percentage of total
    pct = counts / len(speed) * 100.0

    # ── plot ────────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw={'projection': 'polar'})
    else:
        fig = ax.figure

    bar_width = np.radians(sector_width) * 0.9

    for b in range(n_speed_bins):
        bottoms = pct[:, :b].sum(axis=1)
        ax.bar(sector_centres_rad, pct[:, b],
               width=bar_width, bottom=bottoms,
               color=colors[b], label=speed_labels[b],
               edgecolor='white', linewidth=0.4, alpha=0.9)

    # meteorological polar axes: 0=N at top, clockwise
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(np.arange(0, 360, 30)))
    ax.set_xticklabels(['N','30','60','E','120','150',
                         'S','210','240','W','300','330'], fontsize=8)

    # radial axis label
    ax.set_ylabel('Frequency (%)', labelpad=30, fontsize=8)

    ax.set_title(title, fontsize=10, pad=15)
    ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.0),
              fontsize=7, title='Speed', title_fontsize=8)

    return ax



def plot_track_statistics(tracks, dt_hours=3.0):
    """
    Plot lifetime and speed distributions for tracked events.
    """
    lifetimes  = np.array([tr['lifetime']   for tr in tracks])
    speeds     = np.array([tr['mean_speed'] for tr in tracks])
    vel_y_all  = np.concatenate([tr['vel_y_kmh'][1:] for tr in tracks])
    vel_x_all  = np.concatenate([tr['vel_x_kmh'][1:] for tr in tracks])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # lifetime histogram
    axes[0].hist(lifetimes * dt_hours, bins=30, color='steelblue', edgecolor='k', lw=0.5)
    axes[0].set_xlabel('Lifetime (hours)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Track lifetimes  (n={len(tracks)})')
    axes[0].axvline(np.median(lifetimes * dt_hours), color='r',
                    linestyle='--', label=f'Median={np.median(lifetimes*dt_hours):.0f}h')
    axes[0].legend()

    # speed histogram
    axes[1].hist(speeds, bins=30, color='steelblue', edgecolor='k', lw=0.5)
    axes[1].set_xlabel('Mean speed (km/h)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Track mean speeds')
    axes[1].axvline(np.median(speeds), color='r',
                    linestyle='--', label=f'Median={np.median(speeds):.0f} km/h')
    axes[1].legend()

    """
    # velocity components scatter
    axes[2].scatter(vel_x_all, vel_y_all, alpha=0.2, s=5, color='steelblue')
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].axvline(0, color='k', lw=0.5)
    axes[2].set_xlabel('Zonal velocity (km/h)')
    axes[2].set_ylabel('Meridional velocity (km/h)')
    axes[2].set_title('Velocity components')
    """

    # ── panel 3: wind rose ──────────────────────────────────────────────────
    ax3 = axes[2] #fig.add_subplot(1, 3, 3, projection='polar')
    plot_wind_rose_trad(vel_x_all, vel_y_all,
                   n_sectors=12, speed_bin_ms=5.0, ax=ax3,
                   title='Track velocity wind rose')
    plt.tight_layout()


def plot_track_statistics_2(tracks, dt_hours=3.0):
    """
    Plot lifetime histogram, speed histogram, and velocity wind rose.
    """
    lifetimes = np.array([tr['lifetime']   for tr in tracks])
    speeds    = np.array([tr['mean_speed'] for tr in tracks])
    vel_y_all = np.concatenate([tr['vel_y_kmh'][1:] for tr in tracks])
    vel_x_all = np.concatenate([tr['vel_x_kmh'][1:] for tr in tracks])

    fig = plt.figure(figsize=(32, 8))

    # ── panel 1: lifetimes ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.hist(lifetimes * dt_hours, bins=30,
             color='steelblue', edgecolor='k', lw=0.5)
    ax1.set_xlabel('Lifetime (hours)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Track lifetimes  (n={len(tracks)})')
    ax1.axvline(np.median(lifetimes * dt_hours), color='r', linestyle='--',
                label=f'Median={np.median(lifetimes*dt_hours):.0f}h')
    ax1.legend()

    # ── panel 2: mean speeds ────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.hist(speeds, bins=30, color='steelblue', edgecolor='k', lw=0.5)
    ax2.set_xlabel('Mean speed (km/h)')
    ax2.set_ylabel('Count')
    ax2.set_title('Track mean speeds')
    ax2.axvline(np.median(speeds), color='r', linestyle='--',
                label=f'Median={np.median(speeds):.0f} km/h')
    ax2.legend()

    # ── panel 3: wind rose ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(1, 4, 3, projection='polar')
    plot_wind_rose_trad(vel_x_all, vel_y_all,
                   n_sectors=12, speed_bin_ms=5.0, ax=ax3,
                   title='Track velocity wind rose')
    # ── panel 3: wind rose ──────────────────────────────────────────────────
    ax4 = fig.add_subplot(1, 4, 4, projection='polar')
    plot_wind_rose_histo(vel_x_all, vel_y_all,
                   n_sectors=12, speed_bin_ms=5.0, ax=ax4,
                   title='Track velocity wind rose')

    plt.tight_layout()
    plt.show()

def trackarrays(tracks,lat=None,lon=None):


    lifetimes  = np.array([tr['lifetime']   for tr in tracks])
    speeds     = np.array([tr['mean_speed'] for tr in tracks])
    maxlife    = np.max(lifetimes)+1
    ntracks    = len(lifetimes)
    
    
    shape = (ntracks, maxlife)
    
    ixs   = np.full(shape, -1, dtype=int)
    iys   = np.full(shape, -1, dtype=int)
    epwp  = np.full(shape, -1, dtype=int)
    times = np.full(shape, -1, dtype=int)
    if lat is not None and lon is not None:
        lons = np.full(shape, -1, dtype=int)
        lats = np.full(shape, -1, dtype=int)
        
    itrk=0
    for tr in tracks:
        length=len(tr['times'])
        times[itrk,0:length] = np.array( tr['times'] )
        epwp[itrk,0:length]  = np.array( tr['epwp_max'] )
        ixs[itrk,0:length]   = np.array( tr['ix'] )
        iys[itrk,0:length]   = np.array( tr['iy'] )
        if lat is not None and lon is not None:
            lats[itrk,0:length]   = lat[ np.array( tr['iy'][0:length] ) ]
            lons[itrk,0:length]   = lon[ np.array( tr['ix'][0:length] ) ]
            #loop[0:1] = lon[ np.array(tr['ix'][0:1])]             
        itrk=itrk+1

    trx_={'lifetime':lifetimes,'mean_speed':speeds}
    trx_['ixs']      = ixs
    trx_['iys']      = iys
    trx_['epwp_max'] = epwp
    trx_['times']    = times

    if lat is not None and lon is not None:
        trx_['lats']      = lats
        trx_['lons']      = lons

    trx = AttrDict( trx_ )
    return trx

