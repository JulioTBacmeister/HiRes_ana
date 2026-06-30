"""
make_event_cubes.py
====================
Event cube extraction from coarse 1x1 and high-res CESM-MPAS data.
Part of the GW Source Analysis project (DYAMOND / 14km runs)

File A: coarse 1x1 CAM diagnostics, shape (time, ilev, lat, lon) — singleton time
File B: high-res CESM-MPAS w field,  shape (time, ilev, ncol)    — singleton time
        ilev = interface levels, shared between A and B
        ncol = CESM-MPAS unstructured column dimension (~42M cells)

Dependencies: numpy, xarray, scipy.sparse, netCDF4
"""

import numpy as np
import xarray as xr
import netCDF4 as nc
from scipy.sparse import csr_matrix
from pathlib import Path


# =============================================================================
# FILE PATH BUILDERS
# =============================================================================

def _time_stamp(time_event):
    """
    Convert a cftime.DatetimeNoLeap object to 'YYYY-MM-DD-SSSSS' string.
    """
    secs = time_event.hour * 3600 + time_event.minute * 60 + time_event.second
    return f"{time_event.year:04d}-{time_event.month:02d}-{time_event.day:02d}-{secs:05d}"


def get_file_A(time_event,
               base_dir='/glade/derecho/scratch/juliob/archive/cam77_dyamond1_prod1/atm/hist',
               prefix='DynVars_dyamond_fv1x1'):
    """Return path to coarse 1x1 diagnostic file for this event."""
    ts = _time_stamp(time_event)
    return Path(base_dir) / f"{prefix}.{ts}.nc"


def get_file_B(time_event,
               base_dir='/glade/campaign/cgd/amp/juliob/mpasa3p75km/cam77_dyamond1_prod1',
               prefix='w_mpas_prt_dyamond'):
    """Return path to high-res CESM-MPAS file for this event."""
    ts = _time_stamp(time_event)
    return Path(base_dir) / ts / f"{prefix}.{ts}.nc"


# =============================================================================
# REGRIDDER (call once, reuse)
# =============================================================================

def make_regridder(wgtfile):
    """
    Load ESMF weight file and build sparse regrid matrix.

    Returns
    -------
    W      : scipy.sparse.csr_matrix, shape (nlat*nlon, n_src)
    nlat   : int
    nlon   : int
    lat1d  : np.ndarray, shape (nlat,)  — degrees
    lon1d  : np.ndarray, shape (nlon,)  — degrees
    """
    with nc.Dataset(wgtfile) as f:
        row      = f.variables['row'][:] - 1
        col      = f.variables['col'][:] - 1
        S        = f.variables['S'][:]
        dst_dims = f.variables['dst_grid_dims'][:]
        src_dims = f.variables['src_grid_dims'][:]   # ← add this
        yc_b     = f.variables['yc_b'][:]
        xc_b     = f.variables['xc_b'][:]
    
    nlon, nlat = int(dst_dims[0]), int(dst_dims[1])
    n_src      = int(src_dims[0])                    # ← use this
    
    W     = csr_matrix((S, (row, col)), shape=(nlat * nlon, n_src))

    # yc_b/xc_b may be in radians (SCRIP convention) — convert if needed
    if np.abs(yc_b).max() <= np.pi:
        yc_b = np.degrees(yc_b)
        xc_b = np.degrees(xc_b)

    lat2d = yc_b.reshape(nlat, nlon)
    lon2d = xc_b.reshape(nlat, nlon)
    lat1d = lat2d[:, 0]
    lon1d = lon2d[0, :]

    print(f"Regridder ready: {nlat} lat x {nlon} lon")
    print(f"  lat: {lat1d[0]:.3f} to {lat1d[-1]:.3f} deg")
    print(f"  lon: {lon1d[0]:.3f} to {lon1d[-1]:.3f} deg")

    return W, nlat, nlon, lat1d, lon1d


def regrid(field_mpas, W, nlat, nlon):
    """
    Regrid CESM-MPAS unstructured field to regular lat/lon grid.

    Parameters
    ----------
    field_mpas : np.ndarray, shape (..., ncol)
        Last axis must be the unstructured column dimension.
    W          : csr_matrix, shape (nlat*nlon, ncol)

    Returns
    -------
    np.ndarray, shape (..., nlat, nlon)
    """
    orig_shape = field_mpas.shape[:-1]
    flat = field_mpas.reshape(-1, field_mpas.shape[-1])   # (nfields, ncol)
    out  = (W @ flat.T).T                                  # (nfields, nlat*nlon)
    return out.reshape(orig_shape + (nlat, nlon))


# =============================================================================
# LAT/LON INDEX HELPERS
# =============================================================================

def _lat_indices(lat1d, lat_event, half_extent=5.0):
    """
    Return [i0, i1) slice indices and sub-array for lat band around lat_event.
    Clips to array bounds (no wrap for lat).
    """
    latS = lat_event - half_extent
    latN = lat_event + half_extent
    idx  = np.where((lat1d >= latS) & (lat1d <= latN))[0]
    if len(idx) == 0:
        raise ValueError(f"No lat points found around {lat_event:.2f} "
                         f"(lat range: {lat1d.min():.2f} to {lat1d.max():.2f})")
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    return i0, i1, lat1d[i0:i1]


def _lon_indices_wrap(lon1d, lon_event, half_extent=5.0):
    """
    Return indices and sub-array for lon band around lon_event, wrapping 0/360.
    Returned lon_sub is unwrapped (monotonically increasing) for plotting.
    """
    lonW = (lon_event - half_extent) % 360.0
    lonE = (lon_event + half_extent) % 360.0

    if lonW < lonE:
        # no wrap needed
        idx     = np.where((lon1d >= lonW) & (lon1d <= lonE))[0]
        lon_sub = lon1d[idx]
    else:
        # wraps around 0/360 boundary
        idx_hi  = np.where(lon1d >= lonW)[0]
        idx_lo  = np.where(lon1d <= lonE)[0]
        idx     = np.concatenate([idx_hi, idx_lo])
        lon_sub = np.concatenate([lon1d[idx_hi], lon1d[idx_lo] + 360.0])

    if len(idx) == 0:
        raise ValueError(f"No lon points found around {lon_event:.2f}")

    return idx, lon_sub


def _extract_patch(arr, i0_lat, i1_lat, ix_lon):
    """
    Extract a lat/lon patch from a (..., nlat, nlon) array.
    ix_lon may be non-contiguous (wrap case).
    """
    return arr[..., i0_lat:i1_lat, :][..., ix_lon]


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_event_cube(event_idx, ds,
                       W, nlat, nlon, lat1d_hr, lon1d_hr,
                       half_extent=5.0,
                       vars_A=('Upwp', 'Vpwp', 'rho_mpas'),
                       var_B='w_mpas_prt',
                       w_to_midpoints=True,
                       base_dir_A='/glade/derecho/scratch/juliob/archive/cam77_dyamond1_prod1/atm/hist',
                       base_dir_B='/glade/campaign/cgd/amp/juliob/mpasa3p75km/cam77_dyamond1_prod1',
                       verbose=True):
    """
    Extract a spatial cube of fields around a GW event.

    Parameters
    ----------
    event_idx : int
        Integer position into ds (NOT the event_id value).
    ds : xr.Dataset
        Event dataset with coords: index, lat, lon, zlev.
    W, nlat, nlon, lat1d_hr, lon1d_hr
        Regridder outputs from make_regridder().
    half_extent : float
        Half-width of extraction cube in degrees (default ±5°).
    vars_A : tuple of str
        Variables to extract from File A (coarse 1x1).
        Expected shape in file: (time=1, ilev, lat, lon).
    var_B : str
        Variable to extract from File B (CESM-MPAS high-res).
        Expected shape in file: (time=1, ilev, ncol).
    w_to_midpoints : bool
        If True, average w_mpas_prt from ilev interfaces to lev midpoints:
        w_mid[k] = 0.5*(w[k] + w[k+1]), giving shape (nilev-1, ...).
        Set False to keep on interface levels.
    base_dir_A, base_dir_B : str
        Base directories for File A and File B.
    verbose : bool

    Returns
    -------
    cube : dict
        Metadata keys:
            'event_idx', 'event_id', 'itime', 'time_event',
            'lat_event', 'lon_event',
            'lat_A', 'lon_A'   — coarse grid axes for patch
            'lat_hr', 'lon_hr' — high-res grid axes for patch
            'zlev'             — vertical levels from ds
        Data keys (one per var in vars_A):
            e.g. 'Upwp'     : np.ndarray, shape (nilev, nlat_A, nlon_A)
                 'Vpwp'     : np.ndarray, shape (nilev, nlat_A, nlon_A)
                 'rho_mpas' : np.ndarray, shape (nilev, nlat_A, nlon_A)
        High-res key:
            'w_mpas_yx' : np.ndarray, shape (nilev[or nilev-1], nlat_hr, nlon_hr)
    """
    # --- event metadata -----------------------------------------------------
    ev         = ds.isel(index=event_idx)
    time_event = ev['time_event'].values.item()
    lat_event  = float(ev['lat_event'].values)
    lon_event  = float(ev['lon_event'].values)
    event_id   = int(ev['event_id'].values)
    itime      = int(ev['itime'].values)
    ts         = _time_stamp(time_event)

    if verbose:
        print(f"Event {event_idx}  (id={event_id})  t={ts}  "
              f"lat={lat_event:.2f}  lon={lon_event:.2f}")

    # --- file paths ---------------------------------------------------------
    fA = get_file_A(time_event, base_dir=base_dir_A)
    fB = get_file_B(time_event, base_dir=base_dir_B)

    if verbose:
        print(f"  File A: {fA}")
        print(f"  File B: {fB}")

    # --- grid axes ----------------------------------------------------------
    lat_A = ds['lat'].values
    lon_A = ds['lon'].values
    zlev  = ds['zlev'].values

    # coarse grid patch indices
    i0_A, i1_A, lat_sub_A = _lat_indices(lat_A, lat_event, half_extent)
    ix_A,        lon_sub_A = _lon_indices_wrap(lon_A, lon_event, half_extent)

    # high-res grid patch indices
    i0_hr, i1_hr, lat_sub_hr = _lat_indices(lat1d_hr, lat_event, half_extent)
    ix_hr,         lon_sub_hr = _lon_indices_wrap(lon1d_hr, lon_event, half_extent)

    # --- output dict --------------------------------------------------------
    cube = {
        'event_idx'  : event_idx,
        'event_id'   : event_id,
        'itime'      : itime,
        'time_event' : time_event,
        'lat_event'  : lat_event,
        'lon_event'  : lon_event,
        'lat_A'      : lat_sub_A,
        'lon_A'      : lon_sub_A,
        'lat_hr'     : lat_sub_hr,
        'lon_hr'     : lon_sub_hr,
        'zlev'       : zlev,
    }

    # --- File A: coarse 1x1, shape (time=1, ilev, lat, lon) ----------------
    dsA = xr.open_dataset(fA)
    for vname in vars_A:
        if vname not in dsA:
            print(f"  WARNING: {vname} not found in {fA.name}")
            cube[vname] = None
            continue
        arr = dsA[vname].values          # (1, nilev, nlat, nlon)  [or 2D/3D]
        arr = np.squeeze(arr)            # drop all singleton dims
        # now expect (nilev, nlat, nlon) or (nlat, nlon)
        arr_sub = _extract_patch(arr, i0_A, i1_A, ix_A)
        cube[vname] = arr_sub
        if verbose:
            print(f"  {vname}: {arr.shape} → patch {arr_sub.shape}")
    dsA.close()

    # --- File B: CESM-MPAS, shape (time=1, ilev, ncol) ---------------------
    dsB = xr.open_dataset(fB)
    if var_B not in dsB:
        print(f"  WARNING: {var_B} not found in {fB.name}")
        cube['w_mpas_yx'] = None
    else:
        arr_B = dsB[var_B].values        # (1, nilev, ncol)
        arr_B = np.squeeze(arr_B)        # (nilev, ncol)

        if verbose:
            print(f"  {var_B}: raw shape {dsB[var_B].shape} → squeezed {arr_B.shape}")

        # average from ilev interfaces to lev midpoints if requested
        if w_to_midpoints:
            arr_B = 0.5 * (arr_B[:-1, :] + arr_B[1:, :])   # (nilev-1, ncol)
            if verbose:
                print(f"  {var_B}: averaged to midpoints → {arr_B.shape}")

        # regrid all levels at once: (nilev, ncol) → (nilev, nlat_hr, nlon_hr)
        w_yx = regrid(arr_B, W, nlat, nlon)

        # extract patch
        w_sub = _extract_patch(w_yx, i0_hr, i1_hr, ix_hr)
        cube['w_mpas_yx'] = w_sub
        if verbose:
            print(f"  w_mpas_yx: {w_sub.shape}")
    dsB.close()

    return cube