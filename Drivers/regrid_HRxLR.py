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

#from box import Box

def drive(write_file=True, return_dataset=False, verbose=False ):  

    user = os.getenv("USER")  

    #####################################
    # Read YAML and make date string
    file_path = './config_ana.yaml'  # Specify the path to your config file
    config = uc.read_config_yaml( file_path )
    print( config )
    year=int( config['year'] )
    month=int( config['month'] )
    day=config['day']
    hour=config['hour'] 
    print( year, type(year).__name__ , month, type(month).__name__ )
    print( day, type(day).__name__ , hour, type(hour).__name__ )

    case=config['Case']
    if (config['Archive_base'] is None):
        archive_base = f'/glade/derecho/scratch/juliob/archive/'
    else:
        archive_base = config['Archive_base']

    day=handle(day)
    hour=handle(hour)
    print( day, type(day).__name__ , hour, type(hour).__name__ )

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

    print( days_to_do )
    print( hours_to_do )
    
    #####################################
    # Initialize regrid-object library
    RgObLib={}


    RgOb_ne240_x_ne16   = GrU.regrid_object_lib(RgOb=RgObLib, src='ne240pg3', dst='ne16pg3',  RegridMethod='CONSERVE_2ND')
    RgOb_ne16_x_ne240   = GrU.regrid_object_lib(RgOb=RgObLib, src='ne16pg3',  dst='ne240pg3', RegridMethod='BILINEAR')
    RgOb_ne240_x_fv1x1  = GrU.regrid_object_lib(RgOb=RgObLib, src='ne240pg3', dst='fv1x1',    RegridMethod='CONSERVE_2ND')
    RgOb_ne16_x_fv1x1   = GrU.regrid_object_lib(RgOb=RgObLib, src='ne16pg3',  dst='fv1x1',    RegridMethod='BILINEAR')


    lat1R,lon1R = GrU.latlon( grid='fv1x1' )

    
    ######################################################################
    #  Momentum fluxes and other 2nd order moments. 
    #
    #  Algorithm:
    #     regrid (conserve_2) ne240 ==> ne16
    #        uO ==> uOx2
    #        vO ==> vOx2
    #        wO ==> wOx2 (w=omega)
    #     regrid (bilinear) ne16 ==> ne240 
    #        uOx2 ==> uOx2xO
    #        vOx2 ==> vOx2xO
    #        wOx2 ==> wOx2xO
    #
    #  Fields {u,v,w}Ox2xO are regarded as large-scale 
    #  background. Perturbations (on ne240) are then
    #
    #        {up,vp,wp}O = {u,v,w}O - {u,v,w}Ox2xO
    #
    #  A second order moment is then calculated like this. 
    #
    #         upwpO = upO * wpO 
    #
    #     regrid (conserve_2) ne240 ==> ne16
    #          upwpO ==> upwpOx2
    #
    #     regrid (bilinear) ne16 ==> fv1x1
    #          upwpOx2 ==> upwpOx2x1R
    #
    ######################################################################


    regrid_list=['PS','U','V','OMEGA','T']
    for d in days_to_do:
        for h in hours_to_do:
            date_tag = f"{year:04}-{month:02}-{d:02}-{3600*h:05}"
        
            #fin = f'/glade/derecho/scratch/juliob/archive/c153_topfix_ne240pg3_FMTHIST_xic_x02/atm/hist/c153_topfix_ne240pg3_FMTHIST_xic_x02.cam.h1i.{date_tag}.nc'
            fin = f"{archive_base}/{case}/atm/hist/{case}.cam.h1i.{date_tag}.nc"
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

            fout = f"{Bdiro}/{case}.cam.h1i.{date_tag}.nc"

            if( verbose==True):
                print( f"reading {fin}" , flush=True )
                print( f"writing {fout}", flush=True  )
            
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
        
            
            uO = X.U.values
            vO = X.V.values
            wO = X.OMEGA.values
        
            
            uOx2=RgF.Horz(xfld_Src=uO , Src='ne240pg3', Dst='ne16pg3' , RegridObj_In=  RgOb_ne240_x_ne16  ) 
            uOx2xO=RgF.Horz(xfld_Src=uOx2 , Src='ne16pg3' , Dst='ne240pg3', RegridObj_In= RgOb_ne16_x_ne240  ) 
            print( f"finished U" )
            vOx2=RgF.Horz(xfld_Src=vO , Src='ne240pg3', Dst='ne16pg3' , RegridObj_In=  RgOb_ne240_x_ne16  ) 
            vOx2xO=RgF.Horz(xfld_Src=vOx2 , Src='ne16pg3' , Dst='ne240pg3', RegridObj_In= RgOb_ne16_x_ne240  ) 
            print( f"finished V" )
            wOx2=RgF.Horz(xfld_Src=wO , Src='ne240pg3', Dst='ne16pg3' , RegridObj_In=  RgOb_ne240_x_ne16  ) 
            wOx2xO=RgF.Horz(xfld_Src=wOx2 , Src='ne16pg3' , Dst='ne240pg3', RegridObj_In= RgOb_ne16_x_ne240  ) 
            print( f"finished OMEGA" )
        
        
            ##################################################################
            # Now calculate perturbations using coarse-grained=>remapped fields as
            # background
            ##################################################################
            upO = uO - uOx2xO
            vpO = vO - vOx2xO
            wpO = wO - wOx2xO
        
            upwpO= upO * wpO 
            upwpOx2    = RgF.Horz(xfld_Src=upwpO , Src='ne240pg3', Dst='ne16pg3' , RegridObj_In=  RgOb_ne240_x_ne16  ) 
            upwpOx2x1R = RgF.Horz(xfld_Src=upwpOx2 ,  Src='ne16pg3' , Dst='fv1x1', RegridObj_In=  RgOb_ne16_x_fv1x1  ) 
            Dar = xr.DataArray( data=upwpOx2x1R.reshape(nt,nz,ny,nx), 
                                dims=('time','lev','lat','lon',),
                                attrs=dict( long_name='x-monmentum flux',units='m+2 s-2',) ,) 
            Xo['upwp'] = Dar
            if( verbose==True):
                print( f"Finshed with UpWp " , flush=True )

            vpwpO= vpO * wpO 
            vpwpOx2    = RgF.Horz(xfld_Src=vpwpO , Src='ne240pg3', Dst='ne16pg3' , RegridObj_In=  RgOb_ne240_x_ne16  ) 
            vpwpOx2x1R = RgF.Horz(xfld_Src=vpwpOx2 ,  Src='ne16pg3' , Dst='fv1x1', RegridObj_In=  RgOb_ne16_x_fv1x1  ) 
            Dar = xr.DataArray( data=vpwpOx2x1R.reshape(nt,nz,ny,nx), 
                                dims=('time','lev','lat','lon',),
                                attrs=dict( long_name='x-momentum flux',units='m+2 s-2',) ,) 
            Xo['vpwp'] = Dar
            if( verbose==True):
                print( f"Finshed with VpWp " , flush=True )

            for var in regrid_list:
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
                                    attrs=dict( long_name='x-momentum flux',units='m+2 s-2',) ,) 
                Xo[var] = Dar
                if( verbose==True):
                    print( f"Finshed with {var}" , flush=True )
                
            if (write_file==True):
                Xo.to_netcdf( fout )
                print( f"   Wrote {fout} ",flush=True )
                
        
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


