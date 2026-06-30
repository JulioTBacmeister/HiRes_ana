# Basics
import sys
import os


# The usual
from datetime import date
import numpy as np
import xarray as xr

rootdir_ = '../'
if ( rootdir_ not in sys.path ):
    sys.path.append(rootdir_)
    print( f" a path to {rootdir_} added in {__name__} ")

from Utils import MakePressures as MkP
from Utils import time_utils as tuti
from Utils import numerical_utils as nuti
from Utils import MyConstants as Co
import utils as U
import analysis_utils as auti

import time as timelog

Rdair = Co.Rdair()
grav  = Co.grav()


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

def make_rho( te, pint ):
    nt,nz,ny,nx = np.shape( te )
    pmid = 0.5*( pint[:,0:nz,:,:] + pint[:,1:nz+1,:,:] )
    te_int = np.zeros( (nt,nz+1,ny,nx) )
    te_int[:,0,:]=te[:,0,:,:]
    te_int[:,nz,:]=te[:,nz-1,:,:]
    for z in np.arange( start=1,stop=nz ):
        te_int[:,z,:] = 0.5*( te[:,z-1,:] + te[:,z,:] )

    rho  = pmid / (Rdair * te )
    rhoi = pint / (Rdair * te_int )

    return rho, rhoi

def case_defaults( case, nsteps, step_size, start_date, dycore, topofile ):

    subdycore=None
    #####################################
    # Set up lazy defaults
    ####################################
    if (case in ['c153_topfix_ne240pg3_FMTHIST_xic_x02', ]):
        case_x = 'c153_topfix_ne240pg3_FMTHIST_xic_x02'
        base_x = f'/glade/derecho/scratch/juliob/archive/{case_x}/atm/fv1x1/{case_x}.cam.h1i'
        if nsteps is None:
            nsteps = 2*4*31
        if start_date is None:
            start_date = [2004,7,15,0]
        if step_size is None:
            step_size = 6
        if dycore is None:
            dycore = 'SE'
    if (case in ['c153_topfix_ne240pg3_FMTHIST_xic_x03', ]):
        case_x = 'c153_topfix_ne240pg3_FMTHIST_xic_x03'
        base_x = f'/glade/derecho/scratch/juliob/archive/{case_x}/atm/fv1x1/{case_x}.cam.h1i'
        if nsteps is None:
            nsteps = 2*4*31
        if start_date is None:
            start_date = [2007,7,15,0]
        if step_size is None:
            step_size = 6
        if dycore is None:
            dycore = 'SE'
    if (case in ['c153_topfix_ne240pg3_FMTHIST_xic_x04', ]):
        case_x = 'c153_topfix_ne240pg3_FMTHIST_xic_x04'
        base_x = f'/glade/derecho/scratch/juliob/archive/{case_x}/atm/fv1x1/{case_x}.cam.h1i'
        if nsteps is None:
            nsteps = 2*4*31
        if start_date is None:
            start_date = [2011,7,15,0]
        if step_size is None:
            step_size = 6
        if dycore is None:
            dycore = 'SE'
    elif (case in ['cam77_dyamond1','cam77_dyamond1_prod1' ]):
        case_x = 'cam77_dyamond1_prod1'
        base_x=f'/glade/derecho/scratch/juliob/archive/{case_x}/atm/hist/DynVars_dyamond_fv1x1'
        if nsteps is None:
            nsteps = 8*31-1
        if start_date is None:
            start_date = [2016,8,1,3]
        if step_size is None:
            step_size = 3
        if dycore is None:
            dycore = 'MPAS'
    elif (case in ['c124_dyamond1','c124_dyamond1_prod2' ]):
        case_x = 'c124_dyamond1_prod2'
        base_x=f'/glade/derecho/scratch/juliob/archive/{case_x}/atm/hist/DynVars_dyamond_fv1x1'
        if nsteps is None:
            nsteps = 8*31-1
        if start_date is None:
            start_date = [2016,8,1,3]
        if step_size is None:
            step_size = 3
        if dycore is None:
            dycore = 'MPAS'
    elif (case in [ 'xy-rdg-mm-front', ]):
        case_x='xy-rdg-mm-front'    
        base_x=f'/glade/derecho/scratch/juliob/archive/GW_UnitTest/{case_x}/{case_x}.h'
        if nsteps is None:
            nsteps = 4*31
        if start_date is None:
            start_date = [2004,8,1,0]
        if step_size is None:
            step_size = 6
        if dycore is None:
            dycore = 'UnitTest'
            subdycore = 'SE'

    if topofile is None:
        topofile = '/glade/work/juliob/Topo/NCARTopoJTB/cases/fv1x1_Sco100_GrnlAnt/output/fv1x1_gmted2010_modis_bedmachine_nc3000_Laplace0100_noleak_greenlndantarcsgh30fac2.50_20251009.nc'

    print( f"  nsteps={nsteps} " )
    return base_x, nsteps, step_size, start_date, dycore, topofile, subdycore

def read_case( case=None , nsteps=None, step_size=None, start_date=None , dycore=None, topofile=None, super_lat_range=None ):

    tic_overall = timelog.perf_counter()

    # A lat range to reduce memory use
    ###################################
    if super_lat_range is not None:
        lat_south,lat_north=super_lat_range
    else:
        lat_south,lat_north=-90.,90.

    #####################################
    # Set up lazy defaults
    ####################################
    base_x, nsteps, step_size, start_date, dycore, topofile, subdycore = case_defaults( case, nsteps, step_size, start_date, dycore, topofile )

    Topo=xr.open_dataset( topofile )
    Topo=Topo.sel( lat=slice(lat_south, lat_north)) 
    htopo=Topo.PHIS.values/grav
    angll=Topo.ANGLL.values[0,:,:]
    mxdis=Topo.MXDIS.values[0,:,:]
    
    year,month,day,hour= start_date
    start_date_a = f"{year:04d}-{month:02d}-{day:02d}-{hour*3_600:05d}"
    dates=[]
    for n in np.arange( nsteps ):
        date_=f"{year:04d}-{month:02d}-{day:02d}-{hour*3_600:05d}"
        dates.append( date_ )
        if day != 99:
            year,month,day,hour = tuti.increment_hours( [year,month,day,hour], nhours=step_size )

    files_x =[]
    for date in dates:
        f_x = f'{base_x}.{date}.nc'
        files_x.append( f_x )

    print( files_x[0],files_x[-1])
    # return

    tic_begin = timelog.perf_counter()
    X=xr.open_mfdataset( files_x ,  data_vars='different', coords='different', compat='no_conflicts'   )

    print( f" Dataset opened  , dtype of U,V {X.U.values.dtype} {X.V.values.dtype} " )
    X = X.sel( lat=slice(lat_south, lat_north)) 
    print( f" Dataset opened  trimmed. {X.lat.values.min():6.2f} to {X.lat.values.max():6.2f} " )

    tic_end = timelog.perf_counter()
    pTime = f"Read subsetted X w open_mfdata_set  {tic_end - tic_begin:0.4f} seconds"
    print(pTime)

    if (dycore in ['SE','MPAS',]):
        ncdata='none'
        lat=X.lat.values
        lon=X.lon.values
        time=X.time.values
        ######################################################
        #   - MPAS derived fluxes are not scaled by density.
        #   - They are on MPAS interface heights given by zgrid
        #
        #   - CAM-SE ne240 derived fluxes are scaled by a 
        #     stupid 'proxy' density RhoProxy(lev) (in files)
        #   - They are on CAM's layer midlevels PMID
        #=====================================================
        #
        # For clear apples to apples comparison ... it might be 
        # good to interpolate to a fixed pressure or height 
        # and then scale by rho = P_2500 / (R_dair * T_2500)
        
        tic_begin = timelog.perf_counter()
        if ('upwp' in X):
            upwp = X.upwp.values 
        elif ('Upwp' in X):
            upwp = X.Upwp.values 
        
        if ('vpwp' in X):
            vpwp = X.vpwp.values 
        elif ('Vpwp' in X):
            vpwp = X.Vpwp.values 
            
        if ('theta_mpaspwp' in X):
            thpwp = X.theta_mpaspwp.values 
    
        u = X.U.values 
        v = X.V.values
        te = X.T.values

        tic_end = timelog.perf_counter()
        pTime = f"extracted upwp,...U,V,T etc  {tic_end - tic_begin:0.4f} seconds"
        print(pTime)
        
        tic_begin = timelog.perf_counter()
        nt,nz,ny,nx = np.shape( u )  # This is the shape for the duration of this SE,MPAS block
        zeta = np.zeros( ( nt,nz,ny,nx) )
        for t in np.arange( nt ):
            for z in np.arange( nz ):
                zeta[t,z,:,:] = nuti.Sphere_Curl2( f_x=u[t,z,:,:]  , f_y=v[t,z,:,:] , lat=lat, lon=lon , wrap=True, verbose=False)
        tic_end = timelog.perf_counter()
        pTime = f"Vorticity calc  {tic_end - tic_begin:0.4f} seconds"
        print(pTime)
    
        tic_begin = timelog.perf_counter()
        if dycore == 'SE':
            plev, pilev =X.lev.values,X.ilev.values
            zlev, zilev = -7_000. * np.log( plev / 1_000. ) , -7_000. * np.log( pilev / 1_000. )
            pmid,pint,delp = MkP.Pressure( X=X, Gridkey = 'tzyx' )
            ze, zo = MkP.GeopHeight( te, delp, pmid, topo=htopo, Gridkey='tzyx')
            RhoProxy = X.RhoProxy.values
        elif dycore == 'MPAS':
            zlev, zilev = X.lev.values , X.ilev.values
            plev, pilev = 100_000. * np.exp( -zlev / 7_000. ) , 100_000. * np.exp( -zilev / 7_000. ) 
            ze = X.zgrid.values
            pint = X.PINT.values
            #nt,nz,ny,nx = np.shape( pint )
            pmid = 0.5*( pint[:,1:,:,:] + pint[:,0:-1,:,:] )
            if len(np.shape(ze))==3:
                ze = np.tile(ze[None, :, :, :], (nt, 1, 1, 1) )
            zo = 0.5*( ze[:,0:-1,:,:] + ze[:,1:,:,:] )
            RhoProxy = 1.2*plev
            tmp = 0.5*( upwp[:,1:,:,:] + upwp[:,0:-1,:,:] )
            upwp = tmp    
            tmp = 0.5*( vpwp[:,1:,:,:] + vpwp[:,0:-1,:,:] )
            vpwp = tmp    
            if ('theta_mpaspwp' in X):
                tmp = 0.5*( thpwp[:,1:,:,:] + thpwp[:,0:-1,:,:] )
                thpwp = tmp    

        tic_end = timelog.perf_counter()
        pTime = f"rho geopht pint ... calc  {tic_end - tic_begin:0.4f} seconds"
        print(pTime)
    
        tic_begin = timelog.perf_counter()
        epwp = np.sqrt( upwp**2 + vpwp**2 )
    
        rho, rhoi =  make_rho( te, pint )
        
        if dycore == 'SE':
            rho_epwp = epwp
            rho_upwp = upwp
            rho_vpwp = vpwp
        elif dycore == 'MPAS':
            rho_epwp = rho * epwp
            rho_upwp = rho * upwp
            rho_vpwp = rho * vpwp
            if ('theta_mpaspwp' in X):
                rho_thpwp = rho * thpwp
            else:
                rho_thpwp=-9999.
            
        tic_end = timelog.perf_counter()
        pTime = f"rho calc ... epwp  {tic_end - tic_begin:0.4f} seconds"
        print(pTime)

    elif (dycore == 'UnitTest'):
        ncdata = X.ncdata
        nt,nz,ny,nx = X.sizes['time'],  X.sizes['level'],  X.sizes['ny'],  X.sizes['nx']
        j = np.arange(ny).repeat(nx)
        i = np.tile(np.arange(nx), ny)
        
        X2 = (
            X.assign_coords(_j=("ncol", j), _i=("ncol", i))     # mapping ncol -> (j,i)
             .set_index(ncol=("_j", "_i"))
             .unstack("ncol")
             .rename({"_j": "ny", "_i": "nx"})
             .assign_coords(ny=X["lat_R"], nx=X["lon_R"])        # put real coords on axes
        )

        X2['lat']=X['lat_R']
        X2['lon']=X['lon_R']
        X=X2
        lat=X.lat.values
        lon=X.lon.values
        time=X.time.values
        
        if (subdycore == 'SE'):
            hyam, hybm  = X.hyam.values,X.hybm.values
            hyai, hybi  = X.hyai.values,X.hybi.values
            plev, pilev = 1_000.*(hyam+hybm), 1_000.*(hyai+hybi)
            zlev, zilev = -7_000. * np.log( plev / 1_000. ) , -7_000. * np.log( pilev / 1_000. )
            zo,ze = X.ZM.values, X.ZI.values
        elif (subdycore == 'MPAS'):
            zilev = X.ilev.values
            zlev = 0.5*( zilev[0:nz] + zilev[1:nz+1] )
            plev,pilev = 100_000.* np.exp( -zlev / 7_000. ), 100_000. *np.exp( -zilev / 7_000. )
            ze = np.tile(zgrid[None, :, :, :], (nt, 1, 1, 1) )
            zo = 0.5*( ze[:,0:nz,:,:] + ze[:,1:nz+1,:,:] )

        RhoProxy = 1.2 * plev
        u = X.U.values
        v = X.V.values
        te = X.T.values
        zeta = X.ZETA.values
        pmid = X.PMID.values
        pint = X.PINT.values
        rho, rhoi =  make_rho( te, pint )
        ##########################################
        # Grab some parameterized momentum fluxes
        # Moving Mountain for now ...
        ###########################################
        rho_epwp = X.TAU_MOVMTN.values[:,1:,:,:]
        upwp, vpwp, epwp = np.zeros( ( nt,nz,ny,nx) ) , np.zeros( ( nt,nz,ny,nx) ) , np.zeros( ( nt,nz,ny,nx) ) 
        rho_upwp, rho_vpwp = np.zeros( ( nt,nz,ny,nx) ) , np.zeros( ( nt,nz,ny,nx) )

    print(u.shape)
    print(v.shape)
    print(zeta.shape)
    print(zo.shape)

    ### TILT
    tic_begin = timelog.perf_counter()
    tilt=auti.tiltmag(u, v, zeta, zo)
    tic_end = timelog.perf_counter()
    pTime = f"tilting calc  {tic_end - tic_begin:0.4f} seconds"
    print(pTime)


    ### FRONTOGENESIS (AND THETA)
    tic_begin = timelog.perf_counter()
    th=te * (100_000./pmid) ** (2./7.)
    fgf,thx,thy,ux,uy,vx,vy = U.fronto_horz( u, v, th, lon,lat )
    #we wont be using these here
    del thx,thy,ux,uy,vx,vy
    tic_end = timelog.perf_counter()
    pTime = f"frontogenesis calc  {tic_end - tic_begin:0.4f} seconds"
    print(pTime)

    ### STABILITY
    tic_begin = timelog.perf_counter()
    stab=auti.stability(th, zo)
    tic_end = timelog.perf_counter()
    pTime = f"stability calc  {tic_end - tic_begin:0.4f} seconds"
    print(pTime)


    
    A_ = {  'dycore': dycore , 'base_file_name':base_x , 'case':case ,
            'ncdata': ncdata ,
            'topofile':'topofile',
            'start_date': start_date_a , 
            'end_date': dates[-1] , 
            'nsteps': nsteps ,
            'step_size': step_size ,
            'time': time ,
            'ymds': dates , 
            'topo_file': topofile , 
            'lon': lon ,
            'lat': lat ,
            'plev': plev , 
            'zlev': zlev ,
            'pilev': pilev ,
            'zilev': zilev ,
            'RhoProxy': RhoProxy ,
            'pint': pint ,
            'pmid': pmid ,
            'htopo': htopo , 'angll':angll, 'mxdis':mxdis,
            'rho': rho ,
            'rhoi': rhoi ,
            'zo': zo ,                   
            'ze': ze ,                   
            'u': u ,                   
            'v': v ,                   
            'te': te ,
            'th': th ,
            'zeta': zeta ,
            'tilt': tilt ,
            'stab': stab ,
            'fgf': fgf,
            'epwp': epwp ,                   
            'upwp': upwp ,                   
            'vpwp': vpwp ,
            'rho_epwp': rho_epwp ,                   
            'rho_upwp': rho_upwp ,                   
            'rho_vpwp': rho_vpwp ,
            'rho_thpwp': rho_thpwp ,
            #'X': X,
         }

    # Add some 'after thought' quantities ...
    # Precip is tiled to ensure 'fair' counting in ML 
    if ('PRECL' in X):
        precl=X.PRECL.values
        precl = np.tile( precl[:, np.newaxis, :, :],(1,nz,1,1) )
        A_['precl']=precl
    if ('PRECC' in X):
        precc=X.PRECC.values
        precc = np.tile( precc[:, np.newaxis, :, :],(1,nz,1,1) )
        A_['precc']=precc

    
    A = AttrDict( A_ )

    return A


def read_time_avg_case( case=None , nsteps=None, step_size=None, start_date=None , dycore=None, topofile=None ):

    #####################################
    # Set up lazy defaults
    ####################################
    base_x, nsteps, step_size, start_date, dycore, topofile, subdycore = case_defaults( case, nsteps, step_size, start_date, dycore, topofile )

    Topo=xr.open_dataset( topofile )
    htopo=Topo.PHIS.values/grav

    
    year,month,day,hour= start_date
    start_month=month
    start_date_a = f"{year:04d}-{month:02d}-{day:02d}-{hour*3_600:05d}"
    dates=[]
    for n in np.arange( nsteps ):
        date_=f"{year:04d}-{month:02d}-{day:02d}-{hour*3_600:05d}"
        dates.append( date_ )
        year,month,day,hour = tuti.increment_hours( [year,month,day,hour], nhours=step_size )

    files_x =[]
    for date in dates:
        f_x = f'{base_x}.{date}.nc'
        files_x.append( f_x )

    print( files_x[0],files_x[-1])
    # return


    X=xr.open_mfdataset( files_x ,  data_vars='different', coords='different', compat='no_conflicts'   )


    mid_time = X.time.values[len(X.time)//2]
    X_tv=X.mean(dim='time' ).expand_dims(time=[mid_time])
    day,hour=99,0
    date_out=f"{year:04d}-{start_month:02d}-{day:02d}-{hour*3_600:05d}"
    #### return X_tv
    f_out = f'{base_x}.{date_out}.nc'
    print(f"  ... Will now write {f_out} ")
    X_tv.to_netcdf( f_out )
    print(f"  ... Wrote {f_out} ")
    return X_tv