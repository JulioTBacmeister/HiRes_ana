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
from Utils import MakePressures as MkP

import mmt_sources as mms

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

importlib.reload(mms)
importlib.reload(MkP)



# Constants
grav=Co.grav()
Rearth=Co.Rearth()
pi=Co.pi()


def read_in_data(fld='upwp',z=20):

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
                if fld not in ['epwp','vortsrc_v00','vortsrc_v01',]:
                    slice0 = X[fld].values[0,z,:,:]
                elif fld=='epwp':
                    slice0 = np.sqrt( X['upwp'].values[0,z,:,:]**2 + X['vpwp'].values[0,z,:,:]**2 )          
                elif fld=='vortsrc_v00':
                    slice0 = mms.vortsrc_v00( X )        
                elif fld=='vortsrc_v01':
                    slice0 = mms.vortsrc_v01( X )        
                    
                slices.append(slice0)

    ayx = np.stack(slices, axis=0)        # shape = (nt, ny, nx)

    return ayx

################################
def fronto_horz( u, v, th, lon,lat ):




    """
     FV dycore frontogenesis
     !$omp parallel do private (i,j,k, tglat)
      do j=beglatxy, endlatxy

         tglat = grid%sinp(j) / (grid%cosp(j)+1.e-3_r8)

         do k=1, plev
            do i=beglonxy, endlonxy

               frontgf(i,k,j) =                                                                          &
                    - ptx(i,k,j)**2._r8 * (ux(i,k,j) - v3(i,k,j) * tglat / aearth)                          &
                    - pty(i,k,j)**2._r8 * vy(i,k,j)                                                         &
                    - ptx(i,k,j) * pty(i,k,j) * ( vx(i,k,j) + uy(i,k,j) + u3(i,k,j) * tglat / aearth )

            end do
         end do

      end do
    """
    nt,nz,ny,nx=np.shape(th)
    thx = np.zeros( (nt,nz,ny,nx) )
    thy = np.zeros( (nt,nz,ny,nx) )
    ux  = np.zeros( (nt,nz,ny,nx) )
    uy  = np.zeros( (nt,nz,ny,nx) )
    vx  = np.zeros( (nt,nz,ny,nx) )
    vy  = np.zeros( (nt,nz,ny,nx) )
    
    for t in np.arange( nt ):
        for z in np.arange(nz):
            thx[t,z,:,:] , thy[t,z,:,:]  = nuti.Sphere_Grad2_vec(th[t,z,:,:], lat, lon, wrap=True, keep_pole_clamp=True)
            ux[t,z,:,:] ,  uy[t,z,:,:]   = nuti.Sphere_Grad2_vec(u[t,z,:,:], lat, lon, wrap=True, keep_pole_clamp=True)
            vx[t,z,:,:] ,  vy[t,z,:,:]   = nuti.Sphere_Grad2_vec(v[t,z,:,:], lat, lon, wrap=True, keep_pole_clamp=True)

    tglat = np.zeros( (nt,nz,ny,nx) )
    for y in np.arange( ny ):
        tglat[:,:,y,:] = np.tan( lat[y] * pi/180. )
    
    fronto = np.zeros( (nt,nz,ny,nx) )

    fronto = -(thx**2 *( ux - v*tglat/Rearth) )  - (thy**2 * vy) \
            -(thx*thy *( vx + uy + u*tglat/Rearth) )
    

    #return fronto
    return fronto,thx,thy,ux,uy,vx,vy
################################
def grad2_of_dse( dse, lon,lat ):
    
    nt,nz,ny,nx=np.shape(dse)
    gr2 = np.zeros( (nt,nz,ny,nx) )
    
    for t in np.arange( nt ):
        for z in np.arange(nz):
            grx,gry = nuti.Sphere_Grad2_vec(dse[t,z,:,:], lat, lon, wrap=True, keep_pole_clamp=True)
            gr2[t,z,:,:] = grx**2 + gry**2
    
    return gr2
################################
def vorticity( X ):
    
    u=X.U.values
    v=X.V.values
    lat=X.lat.values 
    lon=X.lon.values
    nt,nz,ny,nx=np.shape(u)
    zeta = np.zeros( (nt,nz,ny,nx) )

    print( np.shape( u ) )
    
    for t in np.arange( nt ):
        for z in np.arange(nz):
            zeta[t,z,:,:] = nuti.Sphere_Curl2( f_x=u[t,z,:,:], f_y=v[t,z,:,:], lat=lat, lon=lon, wrap=True )
            #print(t,z)
    
    return zeta
################################
def geopht( X , topo=None ):

    am, bm, ai, bi = X.hyam.values, X.hybm.values, X.hyai.values, X.hybi.values 
    print('am ',am.shape)
    te=X.T.values
    ps=X.PS.values
    print('ps ',ps.shape)

    pmid,pint,delp = MkP.Pressure ( am, bm, ai, bi, ps , p_00=100_000., Gridkey='tzyx' )
    z3e,z3o = MkP.GeopHeight( te, delp, pmid, topo=topo , Gridkey='tzyx')
    
    
    return z3e,z3o
