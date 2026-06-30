"""
gw_05_io.py
===========
Save and load El (list of threshold event objects) to/from NetCDF.

One NetCDF file is written per threshold element, plus a JSON sidecar
for scalar metadata. Files are named:

    {prefix}_thr{i:02d}.nc       — field data for threshold i
    {prefix}_metadata.json       — scalar metadata for all thresholds

Dependencies: numpy, xarray, json, pathlib

Note:
This could/should logically go into event_utils.py, but that is kind of 
growing out of control. So, ...
"""

import json
import numpy as np
import xarray as xr
from pathlib import Path
import pickle
import numpy as np
import numpy as np

# ===============================================================

def pretty_lat(lat):
    """
    Convert latitude array into strings like:
    30N, 15S, Eq
    """

    out = []

    for x in lat:

        if x == 0:
            out.append("Eq")

        elif x > 0:
            out.append(f"{x:g}N")

        else:
            out.append(f"{abs(x):g}S")

    return out


# ===============================================================

def make_base_filename(El0):
    # Make a file name out El contents
        
    if 'thpwp_4D' in El0:
        vers_="_v2"
    else:
        vers_=""

    if 'fld' in El0:
        event_fld = El0.fld
    else:
        event_fld='rho_epwp'

    if El0.peak_footprint != (1,1) :
        footp_ = f"_ftp{El0.peak_footprint[0]}X{El0.peak_footprint[1]}"
    else:
        footp_=""
        
    domain=""
    if 'exclude_orography' in El0:
        if El0.exclude_orography is not None:
            if El0.exclude_orography == True:
                domain='_ocean'
        
    latS,latN=pretty_lat( El0.lat_range )
    zlev_ = f"EvZ{El0.zlev_event/1000:g}km"
    filen = f"{El0.case}_{El0.start_date}-x-{El0.end_date}_{latS}-{latN}{domain}_{zlev_}_{event_fld}{footp_}{vers_}"

    return filen

# ===============================================================
#
# ===============================================================
def write_event_ds(El=None ):

    basedir = '/glade/derecho/scratch/juliob/archive/GW_event_analysis/ds'
    basedir = Path(basedir)
    basedir.mkdir(parents=True, exist_ok=True)

    for El0 in El:
        basefilen = make_base_filename(El0)  
        thresh_ = f"{El0.threshold[0]:.4f}-{El0.threshold[1]:.4g}".replace('.','d').replace('+','') 
        filen = f"{basedir}/{basefilen}_{thresh_}.nc"
        
        ds=El0.ds
        ds.attrs["description"] = "GW events from resolved momentum flux"
        ds.attrs["epwp_definition"] = "sqrt(upwp^2 + vpwp^2)"
        ds.attrs["threshold"] = El0.threshold
        ds.attrs["event_zlevel"] = El0.zlev_event
        ds.attrs["case"] = f"{El0.case}"
        ds.attrs["start_date"] = f"{El0.start_date}"
        ds.to_netcdf( filen )
        print(f"Wrote {filen}")



# =============================================================================
# SIMPLE PICKLE OUTPUT AND LOADING
# =============================================================================

def pickle_write(El):
    # First make a file name out El contents
    basedir = '/glade/derecho/scratch/juliob/archive/GW_event_analysis/PKL'
    basefilen = make_base_filename(El[0])  
    filen = f"{basedir}/{basefilen}.pkl"
    print( f"Wrting pkl file - {filen}" )
    with open(f'{filen}', 'wb') as f:
        pickle.dump(El, f)

    return filen

# =============================================================================
# SCALAR METADATA HELPERS
# =============================================================================

def _scalar_metadata(E):
    """
    Extract JSON-serialisable scalar metadata from one threshold element.
    Numpy scalars and arrays are converted to plain Python types.
    """
    def _jsonify(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        return v

    keys = [
        'case', 'threshold', 'zlev_event', 'lat_range', 'lon_range',
        'exclude_orography', 'N_events', 'Total_events',
        'Frac_of_total_epwp', 'window_tyx', 'peak_footprint',
    ]
    meta = {}
    for k in keys:
        if hasattr(E, k):
            meta[k] = _jsonify(getattr(E, k))
    return meta


# =============================================================================
# SAVE
# =============================================================================

def save_El(El, prefix, outdir='.'):
    """
    Save El to one NetCDF file per threshold plus a JSON metadata sidecar.

    Parameters
    ----------
    El : list of AttrDict
        As returned by make_El().
    prefix : str
        File name prefix, e.g. 'dyamond_SO'.
    outdir : str or Path
        Output directory. Created if it does not exist.

    Returns
    -------
    paths : list of Path
        Paths of all files written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 4D field variables to save — (event, t, z, y, x) unless noted
    field_keys_4D = [
        'zeta_4D', 'tilt_4D', 'fgf_4D',
        'epwp_4D', 'upwp_4D', 'vpwp_4D',
        'u_4D', 'v_4D', 'th_4D', 'htopo_4D',
    ]
    # 2D-in-space fields tiled to 4D — still (event, t, y, x)
    field_keys_tyx = ['precl_4D']

    paths      = []
    all_meta   = []

    for i, E in enumerate(El):
        nc_path = outdir / f'{prefix}_thr{i:02d}.nc'
        print(f"Writing threshold {i}: {nc_path}")

        # ----------------------------------------------------------------
        # 1. Start from the event dataset ds
        #    ds already has dim 'index'; rename to 'event' for clarity
        # ----------------------------------------------------------------
        ds_out = E.ds.rename({'index': 'event'})

        # ----------------------------------------------------------------
        # 2. Add coordinate vectors as 1D coordinates
        # ----------------------------------------------------------------
        ds_out = ds_out.assign_coords(
            zlev=('zlevA', E.zlevA.values
                  if hasattr(E.zlevA, 'values') else np.asarray(E.zlevA)),
        )

        # ----------------------------------------------------------------
        # 3. Add field arrays — infer dims from actual array rank
        # ----------------------------------------------------------------
        dims_5D  = ('event', 't_window', 'zlevA', 'y_window', 'x_window')
        dims_4D  = ('event', 't_window', 'y_window', 'x_window')
        dims_3D  = ('event', 'y_window', 'x_window')

        def _dims_for(arr):
            """Pick the right dim tuple based on array rank."""
            if arr.ndim == 5:
                return dims_5D
            elif arr.ndim == 4:
                return dims_4D
            elif arr.ndim == 3:
                return dims_3D
            else:
                raise ValueError(f"Unexpected array shape {arr.shape} "
                                 f"— don't know how to assign dims")

        all_field_keys = field_keys_4D + field_keys_tyx
        for key in all_field_keys:
            if hasattr(E, key):
                arr = getattr(E, key)
                ds_out[key] = xr.DataArray(arr, dims=_dims_for(arr))
                print(f"  {key}: shape={arr.shape} -> dims={_dims_for(arr)}")
        
        
        
        # ----------------------------------------------------------------
        # 4. Add lat4D / lon4D / time4D as variables
        #    (they vary per event so can't be simple coords)
        # ----------------------------------------------------------------
        if hasattr(E, 'lat4D'):
            ds_out['lat4D'] = xr.DataArray(
                E.lat4D, dims=('event', )) #'t_window', 'y_window', 'x_window'))
        if hasattr(E, 'lon4D'):
            ds_out['lon4D'] = xr.DataArray(
                E.lon4D, dims=('event', )) #'t_window', 'y_window', 'x_window'))
        if hasattr(E, 'time4D'):
            ds_out['time4D'] = xr.DataArray(
                E.time4D, dims=('event', )) #'t_window'))
        if hasattr(E, 'zlevA'):
            ds_out['zlevA'] = xr.DataArray(
                E.zlevA, dims=('zlevA', )) #'t_window'))

        # ----------------------------------------------------------------
        # 5. Add global attributes (scalar metadata)
        # ----------------------------------------------------------------
        meta = _scalar_metadata(E)
        ds_out.attrs.update({k: str(v) for k, v in meta.items()})
        #ds_out.attrs.update({k: v for k, v in meta.items()})
        ds_out.attrs['threshold_index'] = i

        # ----------------------------------------------------------------
        # 6. Write NetCDF
        # ----------------------------------------------------------------
        ds_out.to_netcdf(nc_path)
        paths.append(nc_path)
        all_meta.append({'threshold_index': i, 'nc_file': nc_path.name,
                         **meta})

    # --------------------------------------------------------------------
    # 7. Write JSON sidecar
    # --------------------------------------------------------------------
    json_path = outdir / f'{prefix}_metadata.json'
    with open(json_path, 'w') as f:
        json.dump(all_meta, f, indent=2, default=str)
    paths.append(json_path)
    print(f"Metadata written: {json_path}")

    return paths


# =============================================================================
# LOAD
# =============================================================================

def load_El(prefix, indir='.'):
    """
    Load El from NetCDF files written by save_El().

    Parameters
    ----------
    prefix : str
    indir  : str or Path

    Returns
    -------
    El : list of xr.Dataset
        Each element is a dataset with all fields and metadata attributes.
        Scalar metadata is recoverable from ds.attrs.
    meta : list of dict
        Scalar metadata for each threshold (from JSON sidecar).
    """
    indir = Path(indir)

    # --- load metadata sidecar ---
    json_path = indir / f'{prefix}_metadata.json'
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
    with open(json_path) as f:
        meta = json.load(f)

    El = []
    for m in meta:
        nc_path = indir / m['nc_file']
        if not nc_path.exists():
            raise FileNotFoundError(f"NetCDF file not found: {nc_path}")
        ds = xr.open_dataset(nc_path)
        El.append(ds)
        print(f"Loaded threshold {m['threshold_index']}: "
              f"{m['nc_file']}  "
              f"(N_events={ds.sizes['event']}, "
              f"threshold={m.get('threshold','?')})")

    return El, meta


# =============================================================================
# CONVENIENCE: SUMMARY
# =============================================================================

def summarise_El(El, meta=None):
    """
    Print a summary table of all threshold levels in El.

    Parameters
    ----------
    El   : list of xr.Dataset (from load_El) or list of AttrDict (from make_El)
    meta : list of dict or None — if None, reads from ds.attrs
    """
    print(f"\n{'i':>3}  {'threshold':>18}  {'N_events':>9}  "
          f"{'frac_epwp':>10}  {'window':>12}")
    print('-' * 60)

    for i, E in enumerate(El):
        if meta is not None:
            m = meta[i]
            thr   = m.get('threshold', '?')
            n_ev  = m.get('N_events',  '?')
            frac  = m.get('Frac_of_total_epwp', '?')
            win   = m.get('window_tyx', '?')
        else:
            attrs = E.attrs if hasattr(E, 'attrs') else {}
            thr   = attrs.get('threshold', getattr(E, 'threshold',  '?'))
            n_ev  = attrs.get('N_events',  getattr(E, 'N_events',   '?'))
            frac  = attrs.get('Frac_of_total_epwp',
                              getattr(E, 'Frac_of_total_epwp', '?'))
            win   = attrs.get('window_tyx', getattr(E, 'window_tyx', '?'))

        print(f"{i:>3}  {str(thr):>18}  {str(n_ev):>9}  "
              f"{str(frac):>10}  {str(win):>12}")
    print()



##########################################################################
def write_ds(ds=None, A=None,fld='epwp',extra_info=None,
            fraction_of_total=None, thresh=None, zlev_event=None, zlev=None,
            lat_range=None,lon_range=None ):

    hemi=[]
    for lat in lat_range:
        if lat <0:
            hemi.append('S')
        elif lat>0:
            hemi.append('N')
        elif lat==0:
            hemi.append('')
    
    if extra_info==None:
        extra_tag='' 
    else:
        extra_tag=str(extra_info)
    
    latXlon_ = f"{np.abs(lat_range[0]):02d}{hemi[0]}-{np.abs(lat_range[1]):02d}{hemi[1]}"
    outfile=f"{A.case}_f{fraction_of_total:.1%}{fld}Events_Z{0.001*zlev_event:.0f}km_{A.start_date}-{A.end_date}_{latXlon_}{extra_tag}.nc"
    print( f"writing {outfile}" )

    
    ds.attrs["description"] = "GW events from resolved momentum flux"
    ds.attrs["epwp_definition"] = "sqrt(upwp^2 + vpwp^2)"
    ds.attrs["threshold"] = thresh
    ds.attrs["vertical_level"] = zlev_event
    ds.attrs["source_files"] = f"{A.base_file_name}.%y-%m-%d-%s.nc"
    ds.attrs["start_date"] = f"{A.start_date}"
    ds.attrs["step_size_in_hours"] = f"{A.step_size}"
    ds.to_netcdf( outfile )

