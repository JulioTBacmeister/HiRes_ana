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
from Utils import numerical_utils as nuti
from Utils import MyConstants as Co

#from Drivers import RegridField as RgF
import RegridField as RgF

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

importlib.reload( nuti )


# Constants
grav=Co.grav()

def vortsrc_v00( X ):
    """
    This is intended to reproduce this code. Note steering_level and launch_level are overridden in default
    code by namelist inputs
    !!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine vorticity_flux_src (vorticity , ncol, pverx, alpha_gw_movmtn, vort_src, steering_level, launch_level )
      integer, intent(in) :: ncol,pverx
      real(r8), intent(in) :: vorticity (ncol,pverx)
      real(r8), intent(in) :: alpha_gw_movmtn
    
      real(r8), intent(out) :: vort_src(ncol)
      integer,  intent(out) :: steering_level(ncol), launch_level(ncol)
    
      real(r8) :: scale_factor
      integer  :: k, nlayers
    
      steering_level(:ncol) = pverx - 20
      launch_level(:ncol)   = steering_level -10
    
      scale_factor   = 1.e4_r8 ! scales vorticity amp to u'w' in CLUBB
      !-----------------------------------
      ! Simple average over layers.
      ! Probably can do better
      !-----------------------------------
      nlayers=10
      vort_src(:) =0._r8
      do k = 0, nlayers-1
         vort_src(:) = vort_src(:) + scale_factor * abs( vorticity(:,pverx-k) )
      end do
      vort_src(:) = alpha_gw_movmtn * vort_src(:)/nlayers
    
    end subroutine vorticity_flux_src
    """
    alpha_gw_movmtn = 0.008
    
    u=X.U.values
    v=X.V.values
    lat=X.lat.values 
    lon=X.lon.values
    nt,nz,ny,nx=np.shape(u)
    zeta = np.zeros( (nt,nz,ny,nx) )
    
    for t in np.arange( nt ):
        for z in np.arange(nz):
            zeta[t,z,:,:] = nuti.Sphere_Curl2( f_x=u[t,z,:,:], f_y=v[t,z,:,:], lat=lat, lon=lon, wrap=True )
            #print(t,z)

    src = alpha_gw_movmtn * 1.e4 * np.average( np.abs(zeta[0, nz-11:nz ,:,:]) , axis=0 )
    
    return src

def vortsrc_v01( X ):
    """
    This is intended to reproduce this code. Note steering_level and launch_level are overridden in default
    code by namelist inputs
    !!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine vorticity_flux_src (vorticity , ncol, pverx, alpha_gw_movmtn, vort_src, steering_level, launch_level )
      integer, intent(in) :: ncol,pverx
      real(r8), intent(in) :: vorticity (ncol,pverx)
      real(r8), intent(in) :: alpha_gw_movmtn
    
      real(r8), intent(out) :: vort_src(ncol)
      integer,  intent(out) :: steering_level(ncol), launch_level(ncol)
    
      real(r8) :: scale_factor
      integer  :: k, nlayers
    
      steering_level(:ncol) = pverx - 20
      launch_level(:ncol)   = steering_level -10
    
      scale_factor   = 1.e4_r8 ! scales vorticity amp to u'w' in CLUBB
      !-----------------------------------
      ! Simple average over layers.
      ! Probably can do better
      !-----------------------------------
      nlayers=10
      vort_src(:) =0._r8
      do k = 0, nlayers-1
         vort_src(:) = vort_src(:) + scale_factor * abs( vorticity(:,pverx-k) )
      end do
      vort_src(:) = alpha_gw_movmtn * vort_src(:)/nlayers
    
    end subroutine vorticity_flux_src
    """
    alpha_gw_movmtn = 0.008
    
    u=X.U.values
    v=X.V.values
    lat=X.lat.values 
    lon=X.lon.values
    nt,nz,ny,nx=np.shape(u)
    zeta = np.zeros( (nt,nz,ny,nx) )
    
    for t in np.arange( nt ):
        for z in np.arange(nz):
            zeta[t,z,:,:] = nuti.Sphere_Curl2( f_x=u[t,z,:,:], f_y=v[t,z,:,:], lat=lat, lon=lon, wrap=True )
            #print(t,z)

    src = alpha_gw_movmtn * 1.e4 * np.abs( np.average( zeta[0, nz-25:nz-23 ,:,:] , axis=0 ) )
    
    return src