import sys
import os
workdir_ = '/glade/work/juliob'
if ( workdir_ not in sys.path ):
    sys.path.append(workdir_)
    print( f" a path to {workdir_} added in {__name__} ")

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

from myPythonTools.Utils import constants as co


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


def IwIe( i , nx, wrap=True ):
    iw = i - 1
    if (i==0):
        if (wrap==True):
            iw=nx-1
        else:
            iw=0
    ie = i + 1
    if (i==nx-1):
        if (wrap==True):
            ie=0
        else:
            ie=nx-1
    return iw,ie

def JsJn( j , ny ):
    js = j - 1
    if (j==0):
           js=0
    jn = j + 1
    if (j==ny-1):
        jn=ny-1
    return js,jn

def Sphere_Grad2_vec(f, lat, lon, wrap=True, keep_pole_clamp=True, verbose=False ):
    """
    f:   (ny, nx)
    lat: (ny,)
    lon: (nx,)
    returns: f_x, f_y  (both (ny, nx))
    """
    if verbose==True:
        print( f"this is Sphere_grad_vec in numerical_utils in HiRes_ana")
        print( f"dlat has been adjusted to account for FUCKED UP CRAPPY fv1x1 SCRIP FILE IN CESMDATA INPUTs!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
    Pi  = co.pi()
    R_e = co.Rearth()

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    f   = np.asarray(f)
    ny, nx = f.shape

    rlat = (Pi/180.0) * lat
    rlon = (Pi/180.0) * lon

    ##################################
    # AAAAAAAAAAAAHhhhhhhhhhhhhhh!!!!!!!!!!
    ##################################
    dlat = rlat[3] - rlat[1]
    
    dlon = rlon[2] - rlon[0]

    coslat = np.cos(rlat)[:, None]  # (ny,1) for broadcasting

    # ---------- d/dlon ----------
    if wrap:
        # central diff with periodic wrap
        dfdlon = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / dlon
    else:
        # central interior, one-sided edges
        dfdlon = np.empty_like(f)
        dfdlon[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / dlon
        dfdlon[:, 0]    = (f[:, 1]  - f[:, 0])  / (rlon[1] - rlon[0])
        dfdlon[:, -1]   = (f[:, -1] - f[:, -2]) / (rlon[-1] - rlon[-2])

    f_x = (1.0 / (R_e * coslat)) * dfdlon

    # ---------- d/dlat ----------
    dfdlat = np.empty_like(f)
    dfdlat[1:-1, :] = (f[2:, :] - f[:-2, :]) / dlat

    if keep_pole_clamp:
        # matches your JsJn behavior: js=0,jn=1 at south edge -> (f1-f0)/dlat in your formula?
        # BUT your code uses (f[jn]-f[js]) with js=0, jn=0 at j=0 => 0.
        # So clamp to zero at the ends:
        dfdlat[0, :]  = 0.0
        dfdlat[-1, :] = 0.0
    else:
        # one-sided ends (often what you actually want)
        dfdlat[0, :]  = (f[1, :]  - f[0, :])  / (rlat[1] - rlat[0])
        dfdlat[-1, :] = (f[-1, :] - f[-2, :]) / (rlat[-1] - rlat[-2])

    f_y = (1.0 / R_e) * dfdlat

    return f_x, f_y


def Sphere_Grad2( f, lat, lon, wrap=True ):
    import numpy as np

    Pi  = co.pi()
    R_e = co.Rearth()

    nx,ny = len( lon ), len( lat )
    
    rlat = (Pi/180.)*lat
    rlon = (Pi/180.)*lon
    dlat=rlat[3]-rlat[1]
    dlon=rlon[2]-rlon[0]

    f_x = np.zeros( (ny,nx) )
    for j in np.arange( ny ):
        coslat = np.cos( rlat[j] )
        for i in np.arange( nx ):
            iw,ie = IwIe(i,nx,wrap=wrap)
            f_x[j,i] = (1./(R_e * coslat))*( f[j,ie]-f[j,iw] ) / dlon
    f_y = np.zeros( (ny,nx) )
    for j in np.arange( ny ):
        for i in np.arange( nx ):
            js,jn = JsJn(j,ny)
            f_y[j,i] = (1./ R_e)*( f[jn,i]-f[js,i] ) / dlat

    return f_x, f_y

def Sphere_Div2( f_x, f_y, lat, lon, wrap=True ):
    import numpy as np

    Pi  = co.pi()
    R_e = co.Rearth()

    nx,ny = len( lon ), len( lat )
    
    rlat = (Pi/180.)*lat
    rlon = (Pi/180.)*lon
    dlat=rlat[3]-rlat[1]
    dlon=rlon[2]-rlon[0]
    coslat = np.cos( rlat )

    divf=np.zeros( (ny, nx ))
    for j in np.arange( ny ):
        for i in np.arange( nx ):
            iw,ie = IwIe(i,nx,wrap=wrap)
            js,jn = JsJn(j,ny)
            divf[j,i] = (1./(R_e * coslat[j]))*( 
                        ( f_x[j,ie]-f_x[j,iw] ) / dlon
                    +   ( coslat[jn]*f_y[jn,i] - coslat[js]*f_y[js,i] )/dlat )

    divf[-1,:]=0.
    divf[0,:]=0.

    return divf



def Sphere_Curl2( f_x, f_y, lat, lon, wrap=True, verbose=False):
    ##################################################
    # Curl of a vector field f=[f_x,f_y] in latlon
    # coords:
    #        Zeta ~ d(f_y)/dx - d(f_x)/dy 
    #
    #-------------------------------------------------
    # Inputs need to be 2D (ny,nx) !!
    ##################################################
    import numpy as np

    if verbose==True:
        print( f"this is Sphere_curl2 in numerical_utils in HiRes_ana")
        print( f"dlat has been adjusted to account for FUCKED UP CRAPPY fv1x1 SCRIP FILE IN CESMDATA INPUTs!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
    
    Pi  = co.pi()
    R_e = co.Rearth()

    nx,ny = len( lon ), len( lat )
    
    rlat = (Pi/180.)*lat
    rlon = (Pi/180.)*lon
    
    #########################################################
    # This is extremely non-ideal, but the fv1x1 scrip file in CESMDATA is
    # totally FOOBARRED and we are just going to have live with it. Southmost lat 
    # is weird, and northwrad is weird but at least uniform
    #########################################################
    dlat=rlat[3]-rlat[1]

    dlon=rlon[2]-rlon[0]
    coslat = np.cos( rlat )

    curlf_z = np.zeros( (ny, nx ))

    coslat_jn = np.roll(coslat, -1)
    coslat_js = np.roll(coslat, 1)
    
    # Compute the curl component using vectorized operations
    curlf_z = (1.0 / (R_e * coslat[:, np.newaxis])) * (
        (np.roll(f_y, -1, axis=1) - np.roll(f_y, 1, axis=1)) / dlon
        - (coslat_jn[:, np.newaxis] * np.roll(f_x, -1, axis=0) - coslat_js[:, np.newaxis] * np.roll(f_x, 1, axis=0)) / dlat
        )

    #---------------------------------------------
    # The code above replaces the loop below and
    # is at least 100x faster ???!!!! WTF??
    # Thanks again ChatGPT ...
    """
    for j in np.arange( ny ):
        for i in np.arange( nx ):
            iw,ie = IwIe(i,nx,wrap=wrap)
            js,jn = JsJn(j,ny)
            curlf_z[j,i] = (1./(R_e * coslat[j]))*( 
                     1.0* ( f_y[j,ie]-f_y[j,iw] ) / dlon
                    -   ( coslat[jn]*f_x[jn,i] - coslat[js]*f_x[js,i] )/dlat )
    """

    # Handle boundaries if needed (depending on how your boundary conditions are set)
    # For example, if you have periodic boundary conditions, np.roll takes care of it.
    # If you have different boundary conditions, you may need to handle them separately.
    #curlf_z[0,:]=0.
    #curlf_z[ny-1,:]=0.
    curlf_z[0:1 ,:]=0.
    curlf_z[ny-2:ny-1,:]=0.
    
    return curlf_z



def Sphere_Lap2( f , lat, lon, wrap=True ):

    f_x,f_y = Sphere_Grad2( f=f, lon=lon,lat=lat,wrap=wrap )
    lapf    = Sphere_Div2( f_x=f_x, f_y=f_y, lon=lon,lat=lat,wrap=wrap )
    
    return lapf


#---------------------------------------------------------------
# Function to perform quick coarse graining in lieu of more 
# more accurate conservative remapping
#---------------------------------------------------------------
def coarse_grain(array, block_size, lsum=False):
    # Get the shape of the input array
    m, n = array.shape

    lmean=(not lsum)
    
    # Ensure the input array can be evenly divided into blocks
    assert m % block_size == 0 and n % block_size == 0, "Array dimensions must be divisible by the block size"
    
    # Reshape the array to a 4D array (blocks)
    reshaped_array = array.reshape(m // block_size, block_size, n // block_size, block_size)
    
    if (lmean==True):
        # Compute the mean of each block
        coarse_grained_array = reshaped_array.mean(axis=(1, 3))
    if (lsum==True):
        # Compute the mean of each block
        coarse_grained_array = reshaped_array.sum(axis=(1, 3))
    
    return coarse_grained_array


def bilinear_regrid(y_src=None, x_src=None, data_src=None, y_tgt=None, x_tgt=None):
    """
    Regrid 2D data using bilinear interpolation on a logically rectangular grid.

    Parameters:
    - y_src (1D): Source Y coordinates (e.g., lat), can be non-uniform
    - x_src (1D): Source X coordinates (e.g., lon), must be increasing
    - data_src (2D): Source data [Y, X]
    - y_tgt (2D): Target Y coordinates (meshgrid-style)
    - x_tgt (2D): Target X coordinates (meshgrid-style)

    Returns:
    - data_tgt (2D): Interpolated data on the target grid
    """
    # Create interpolator (note: Y first, then X)
    interp_func = RegularGridInterpolator(
        (y_src, x_src), data_src,
        method='linear', bounds_error=False, fill_value=np.nan
    )

    # Stack target points as [N, 2] array for evaluation
    tgt_points = np.column_stack((y_tgt.ravel(), x_tgt.ravel()))

    # Interpolate and reshape
    data_tgt = interp_func(tgt_points).reshape(y_tgt.shape)
    return data_tgt

    

def ZsY_x_ZtY( y_src, x_src, data_src, y_tgt, x_tgt,
                    kind='linear', bounds_error=False, fill_value=np.nan):
    """
    Perform 2D regridding via sequential 1D interpolations.
    This routine assumes the X grids are not dependent on Y, 
    while the Y grids may depend on X, e.g., pressure altitudes 
    could be a function of latitude. The x_src and x_tgt inputs 
    could actually be 1D vectors, but are 2D here for consistency(?),
    readability(?), uniformity(?) ... ???
    
    Parameters:
    - y_src (2D): source Y-axis (e.g., height/pressure)
    - x_src (2D): source X-axis (e.g., latitude). Note, this coordinate is assumed 
      to be 'regular', i.e., not dependent on y_src.
    - data_src (2D): source data with shape as {x,y}_src
    - y_tgt (2D): target Y-axis
    - x_tgt (2D): target X-axis. Note, this coordinate is assumed 
      to be 'regular', i.e., not dependent on y_tgt.
    - kind: 'linear', 'nearest', 'cubic', etc.
    - bounds_error: raise error if outside range (default: False)
    - fill_value: value to use outside domain (default: NaN)
    
    Returns:
    - data_tgt (2D): shape (len(y_tgt), len(x_tgt))
    """

    
    ny_tgt, nx_tgt = np.shape ( x_tgt ) 
    ny_src, nx_src = np.shape ( x_src ) 


    # Create intermediate 2D X grid that has
    # x_tgt values on y_src levels. Not really 
    # needed, could use x_tgt as a 1D array.
    x_tgt_on_y_src = np.empty( ( ny_src, nx_tgt ))
    for i in range( ny_src ):
        x_tgt_on_y_src[i,:] = x_tgt[0,:]
    
    # First interpolate along horz (X) for each layer
    data_on_y_src_x_tgt = np.empty( ( ny_src, nx_tgt ))
    for i in range( ny_src ):
        f = interp1d(x_src[i,:] , data_src[i, :], kind=kind,
                     bounds_error=bounds_error, fill_value=fill_value)
        data_on_y_src_x_tgt[i, :] = f( x_tgt_on_y_src[i,:] )
      
    # Now interpolate in the vertical (Y) for each column
    data_tgt = np.empty( ( ny_tgt, nx_tgt)  )
    for i in range( nx_tgt ):
        f = interp1d(y_src[:,i] , data_on_y_src_x_tgt[:,i], kind=kind,
                     bounds_error=bounds_error, fill_value=fill_value)
        data_tgt[:,i] = f( y_tgt[:,i] )
    
    return data_tgt
