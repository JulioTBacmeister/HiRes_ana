#!/usr/bin/env python
################################################
# New style 
# ###############################################
import sys
import os


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

def drive(write_file=True, return_dataset=False, verbose=True ):  

    user = os.getenv("USER")  

    #####################################
    # Initialize config
    config = uc.initialize()
    print( config )
    #####################################
    # Read YAML and make date string
    #########
    file_path = './config_wavecube.yaml'  # Specify the path to your config file
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
    
    write_to_file = config['write_to_file']

    day=handle(day)
    hour=handle(hour)
    #print( day, type(day).__name__ , hour, type(hour).__name__ )

    if (day==99): 
        #days_to_do=np.arange( uti.days_in_month(year,month) )+1
        days_to_do = np.arange( start=24,stop=32 ) # KLUGE !!!!! REMOVE!!!!
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

    print( f"Will do days={days_to_do}", flush=True )
    print( f"Will do hours={hours_to_do}", flush=True )

    #####################################
    # Initialize regrid-object library
    RgObLib={}


    RgOb_ne240_x_ne16   = GrU.regrid_object_lib(RgOb=RgObLib, src='ne240pg3', dst='ne16pg3',  RegridMethod='CONSERVE_2ND')
    RgOb_ne16_x_ne240   = GrU.regrid_object_lib(RgOb=RgObLib, src='ne16pg3',  dst='ne240pg3', RegridMethod='BILINEAR')

    RgOb_ne240_x_latlonOxO  = GrU.regrid_object_lib(RgOb=RgObLib, src='ne240pg3', dst='latlonOxO',    RegridMethod='BILINEAR')
    
    #RgOb_ne240_x_fv1x1  = GrU.regrid_object_lib(RgOb=RgObLib, src='ne240pg3', dst='fv1x1',    RegridMethod='CONSERVE_2ND')
    #RgOb_ne16_x_fv1x1   = GrU.regrid_object_lib(RgOb=RgObLib, src='ne16pg3',  dst='fv1x1',    RegridMethod='BILINEAR')


    lat1R,lon1R = GrU.latlon( grid='fv1x1' )
    latOR,lonOR = GrU.latlon( grid='latlonOxO' )

    
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
            
            if Path(fin).is_file():            
                print( f"Processing {fin} ", flush=True )
                X=xr.open_dataset( fin )
    
                ##################################################################
                # Set up dataset for regridded latlonOxO "OR" data
                ##################################################################
                if (config["Output_abs_dir"] is None):
                    Bdiro=f"{archive_base}/{case}/atm/WaveCube"  #{case}.cam.h1i.{date_tag}.nc"
                else:
                    Bdiro=f"{config["Output_abs_dir"]}"
                #######
                os.makedirs( Bdiro , exist_ok=True )
    
                fout = f"{Bdiro}/{case}.cam.h1i.{date_tag}.nc"
                print( f"Will write here {fout} ", flush=True )
    
                if( verbose==True):
                    print( f"reading {fin}" , flush=True )
                    print( f"writing {fout}", flush=True  )
                
                coords = dict( 
                    lon  = ( ["lon"],lonOR ),
                    lat  = ( ["lat"],latOR ),
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

                rhoprx = 1.2* lev/1_000.
                Dar = xr.DataArray( data=rhoprx, 
                                    dims=('lev',),
                                    attrs=dict( long_name='Proxy-air-density',units='kg m-3',) ,) 
                Xo['RhoProxy'] = Dar
                
                
                uO = X.U.values
                vO = X.V.values
                wO = X.OMEGA.values
                tO = X.T.values
                psO = X.PS.values

                # Scale omega to something like w
                for k in np.arange( nz ):
                    wO[:,k,:] = (-1./(grav*rhoprx[k]))*wO[:,k,:]
            
                
                ##################################################################
                # Now calculate backgrounds using coarse-grained fields as
                ##################################################################
                uOx2=RgF.Horz(xfld_Src=uO , Src='ne240pg3', Dst='ne16pg3' , RegridObj_In=  RgOb_ne240_x_ne16  ) 
                uOx2xO=RgF.Horz(xfld_Src=uOx2 , Src='ne16pg3' , Dst='ne240pg3', RegridObj_In= RgOb_ne16_x_ne240  ) 
                print( f"finished smoothing U" , flush=True )
                vOx2=RgF.Horz(xfld_Src=vO , Src='ne240pg3', Dst='ne16pg3' , RegridObj_In=  RgOb_ne240_x_ne16  ) 
                vOx2xO=RgF.Horz(xfld_Src=vOx2 , Src='ne16pg3' , Dst='ne240pg3', RegridObj_In= RgOb_ne16_x_ne240  ) 
                print( f"finished smoothing V" , flush=True  )
                wOx2=RgF.Horz(xfld_Src=wO , Src='ne240pg3', Dst='ne16pg3' , RegridObj_In=  RgOb_ne240_x_ne16  ) 
                wOx2xO=RgF.Horz(xfld_Src=wOx2 , Src='ne16pg3' , Dst='ne240pg3', RegridObj_In= RgOb_ne16_x_ne240  ) 
                print( f"finished smoothing OMEGA" , flush=True )
                tOx2=RgF.Horz(xfld_Src=tO , Src='ne240pg3', Dst='ne16pg3' , RegridObj_In=  RgOb_ne240_x_ne16  ) 
                tOx2xO=RgF.Horz(xfld_Src=tOx2 , Src='ne16pg3' , Dst='ne240pg3', RegridObj_In= RgOb_ne16_x_ne240  ) 
                print( f"finished smoothing T" , flush=True )
                psOx2=RgF.Horz(xfld_Src=psO , Src='ne240pg3', Dst='ne16pg3' , RegridObj_In=  RgOb_ne240_x_ne16  ) 
                psOx2xO=RgF.Horz(xfld_Src=psOx2 , Src='ne16pg3' , Dst='ne240pg3', RegridObj_In= RgOb_ne16_x_ne240  ) 
                print( f"finished smoothing PS" , flush=True )
            
        
                ##################################################################
                # Now calculate perturbations using coarse-grained=>remapped fields as
                # background
                ##################################################################
                upO = uO - uOx2xO
                vpO = vO - vOx2xO
                wpO = wO - wOx2xO
                tpO = tO - tOx2xO

                #######################################################
                # Now place backgournd and perts on latlonOxO grid
                #######################################################
                # - U 
                uOx2xOxOR = RgF.Horz(xfld_Src=uOx2xO ,  Src='ne240pg3' , Dst='latlonOxO', RegridObj_In=  RgOb_ne240_x_latlonOxO  ) 
                Dar = xr.DataArray( data=uOx2xOxOR , 
                                    dims=('time','lev','lat','lon',),
                                    attrs=dict( long_name='U background',units='m s-1',) ,) 
                Xo['Ubk'] = Dar

                upOxOR = RgF.Horz(xfld_Src=upO ,  Src='ne240pg3' , Dst='latlonOxO', RegridObj_In=  RgOb_ne240_x_latlonOxO  ) 
                Dar = xr.DataArray( data=upOxOR , 
                                    dims=('time','lev','lat','lon',),
                                    attrs=dict( long_name='U pert',units='m s-1',) ,) 
                Xo['up'] = Dar
                if( verbose==True):
                    print( f"Finshed with U " , flush=True )

                # - V 
                vOx2xOxOR = RgF.Horz(xfld_Src=vOx2xO ,  Src='ne240pg3' , Dst='latlonOxO', RegridObj_In=  RgOb_ne240_x_latlonOxO  ) 
                Dar = xr.DataArray( data=vOx2xOxOR , 
                                    dims=('time','lev','lat','lon',),
                                    attrs=dict( long_name='V background',units='m s-1',) ,) 
                Xo['Vbk'] = Dar

                vpOxOR = RgF.Horz(xfld_Src=vpO ,  Src='ne240pg3' , Dst='latlonOxO', RegridObj_In=  RgOb_ne240_x_latlonOxO  ) 
                Dar = xr.DataArray( data=vpOxOR , 
                                    dims=('time','lev','lat','lon',),
                                    attrs=dict( long_name='V pert',units='m s-1',) ,) 
                Xo['vp'] = Dar
                if( verbose==True):
                    print( f"Finshed with V " , flush=True )

                # - W 
                wOx2xOxOR = RgF.Horz(xfld_Src=wOx2xO ,  Src='ne240pg3' , Dst='latlonOxO', RegridObj_In=  RgOb_ne240_x_latlonOxO  ) 
                Dar = xr.DataArray( data=wOx2xOxOR , 
                                    dims=('time','lev','lat','lon',),
                                    attrs=dict( long_name='W background',units='m s-1',) ,) 
                Xo['Wbk'] = Dar

                wpOxOR = RgF.Horz(xfld_Src=wpO ,  Src='ne240pg3' , Dst='latlonOxO', RegridObj_In=  RgOb_ne240_x_latlonOxO  ) 
                Dar = xr.DataArray( data=wpOxOR , 
                                    dims=('time','lev','lat','lon',),
                                    attrs=dict( long_name='W pert',units='m s-1',) ,) 
                Xo['wp'] = Dar
                if( verbose==True):
                    print( f"Finshed with W " , flush=True )

                # - T 
                tOx2xOxOR = RgF.Horz(xfld_Src=tOx2xO ,  Src='ne240pg3' , Dst='latlonOxO', RegridObj_In=  RgOb_ne240_x_latlonOxO  ) 
                Dar = xr.DataArray( data=tOx2xOxOR , 
                                    dims=('time','lev','lat','lon',),
                                    attrs=dict( long_name='T background',units='K',) ,) 
                Xo['Tbk'] = Dar

                tpOxOR = RgF.Horz(xfld_Src=tpO ,  Src='ne240pg3' , Dst='latlonOxO', RegridObj_In=  RgOb_ne240_x_latlonOxO  ) 
                Dar = xr.DataArray( data=tpOxOR , 
                                    dims=('time','lev','lat','lon',),
                                    attrs=dict( long_name='T pert',units='K',) ,) 
                Xo['tp'] = Dar
                if( verbose==True):
                    print( f"Finshed with W " , flush=True )

                # - PS 
                psOx2xOxOR = RgF.Horz(xfld_Src=psOx2xO ,  Src='ne240pg3' , Dst='latlonOxO', RegridObj_In=  RgOb_ne240_x_latlonOxO  ) 
                Dar = xr.DataArray( data=psOx2xOxOR , 
                                    dims=('time','lat','lon',),
                                    attrs=dict( long_name='background surf. press.',units='Pa',) ,) 
                Xo['PSbk'] = Dar

                psOxOR = RgF.Horz(xfld_Src=psO ,  Src='ne240pg3' , Dst='latlonOxO', RegridObj_In=  RgOb_ne240_x_latlonOxO  ) 
                Dar = xr.DataArray( data=psOxOR , 
                                    dims=('time','lat','lon',),
                                    attrs=dict( long_name='surface press.',units='Pa',) ,) 
                Xo['PS'] = Dar
                if( verbose==True):
                    print( f"Finshed with PS " , flush=True )

                if( write_to_file==True):
                    Xo.to_netcdf( fout )
                    RC=0
                    #return RC
                else:
                    return Xo


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


