#!/usr/bin/env python
################################################
# New style 
# ###############################################

# Results for 3.75 DYAMOND runs ...
# In /glade/campaign/cesm/km-scale/archive/
#
# cam77_dyamond1_prod1
# c124_dyamond1_prod2

import subprocess
import os
import xarray as xr
import numpy as np
import update_config as uc
from pathlib import Path

def drive():  
    
    #####################################
    # Initialize config
    config = uc.initialize()
    print( config )
    #####################################
    # Read YAML and make date string
    #########
    file_path = './config_mpas_ana.yaml'  # Specify the path to your config file
    config = uc.read_config_yaml( file_path )
    print( config )
    year=int( config['year'] )
    month=int( config['month'] )
    day=config['day']
    hour=config['hour'] 
    case=config['Case']
    hs_in = config['hsPat'] #f"h2i"  #f"h1i"

    UHRtag = config['UHRtag']

    print( f"Year={year},  Month={month}", flush=True )

    date_=f"{year:04d}-{month:02d}-{day:02d}-{hour*3_600:05d}"
    print( f"Processing MPAS 3.75km for {date_}", flush=True)
    SrcDir = f"/glade/campaign/cesm/km-scale/archive/{case}/atm/hist"

    WrkDir = f"/glade/campaign/cgd/amp/juliob/mpasa3p75km/{case}/{date_}"
    os.makedirs( WrkDir , exist_ok=True )

    TimeInvDir = f"/glade/campaign/cgd/amp/juliob/mpasa3p75km/{case}/TimeInvariant"
    os.makedirs( TimeInvDir , exist_ok=True )

    infile = f"{SrcDir}/{case}.cam.{hs_in}.{date_}.nc"


    ###############################################
    # Step 1:
    #   Extract var and make background
    # Step 2:
    #   Make perturbations
    # Step 3:
    #    Calculate flux quadratics wp*(U,T, ... ) on mpasa3p75km
    # Step 4:
    #   Coarse grain flux quadratics to ne16 and then project to fv1x1
    # Step 5:
    #   Make other fileds on fv1x1 grid
    # Step 6:
    #   Make some dyn fields on a UHR (lat-lon) regional grid
    # Step 7:
    #    Calculate 2nd set quadratics Tp2, wp2 on mpasa3p75km
    # Step 8:
    #   Coarse grain 2nd set quadratics to ne16 and then project to fv1x1
    # Step 9:
    #   Combine all fv1x1 files
    # Step 10:
    #   Regrid time-invariant grid files like zgrid to fv1x1
    #############################################################
    #            1        2       3       4       5       6       7       8       9       10
    #do_steps =[ True ,  True  , True ,  True  , True  ,  False , True  , True  , True  , False , ]  # Basic processing
    #do_steps =[ False , False  , False , False , False , False , False , False , True  , False , ] 
    #do_steps =[ True , False  , False , False , False ,  False , False , False , False , False , ] 

    #do_steps =[ True , False  , False , False , True , False , False , False , False , False , ] 
    #step1_B_make_background = False

    #            1        2       3       4       5       6       7       8       9       10
    do_steps = config['do_steps'] #[ False , False  , False , False , False , True , False , False , True  , False , ] 
    
    step=1
    if ( do_steps[ step-1 ] == True ):
        ###############################################
        # Step 1:
        #   Extract var and make background
        ###############################################
        vars = ['U','V','T','Q','w_mpas','theta_mpas','rho_mpas','PINT',]
        vars = ['PRECC','PRECL',]
        #vars = ['w_mpas', 'theta_mpas','rho_mpas', ]

        for var in vars:

            if ( (case=='cam77_dyamond1_prod1' ) and (var in ['w_mpas','theta_mpas','rho_mpas',]) ):
                varx = var.split("_")[0]
                print( f" ... renaming {var} to {varx} ")
            else:
                varx = var

            tmp1    = f"{WrkDir}/{var}_dyamond.nc"
            tmp1x   = f"{WrkDir}/{varx}_dyamond.nc"

            # Need extra steps for cam77 dyamond run which staved w,rho, and theta
            # rather than {}_mpas ... Also these (t,z,nCells) ...
            
            cmds = [
                # extract variable
                f"ncks -O -v {varx},lat,lon {infile} {tmp1x}",
               ]

            
            # DISABLE IF 'fld' FILES ARE ALREADY EXTRACTED
            for c in cmds:
                print("Running:", c, flush=True )
                result = subprocess.run(c, shell=True)
            
                if result.returncode != 0:
                    raise RuntimeError(f"Command failed: {c}")
            """
            for c in cmds:
                print ( f" SKIPPED {c} because already there ", flush=True )
            """            
            if ( (case=='cam77_dyamond1_prod1') and (var in ['w_mpas','theta_mpas','rho_mpas',]) ):
                # This will create tmp1 from tmp1x 
                status = fix_cam77_run( tmp1x, tmp1, varx, var )

            if step1_B_make_background == True:
                #######################################################################
                # Make background by coarsening to ne16pg3 and prolongin to mpasa3p75
                ######################################################################
                tmp2   = f"{WrkDir}/{var}_dyamond_ne16pg3.nc"
                outfile= f"{WrkDir}/{var}_dyamond_ne16pg3_mpasa3p75.{date_}.nc"
    
                cmds = [
                    # conservative remap mpasa3p75km -> ne16pg3
                    "ncremap -m /glade/work/juliob/HiRes_ana_dev/Drivers/mpasa3p75_TO_ne16pg3_cnsrv.nc "
                    f"{tmp1} {tmp2}",
                
                    # bilinear remap ne16pg3 -> mpasa3p75km
                    "ncremap -m /glade/work/juliob/HiRes_ana_dev/Drivers/ne16pg3_TO_mpasa3p75_bilin.nc "
                    f"{tmp2} {outfile}",
                    ]
                
                for c in cmds:
                    print("Running:", c, flush=True )
                    result = subprocess.run(c, shell=True)
                
                    if result.returncode != 0:
                        raise RuntimeError(f"Command failed: {c}")
            
        print("Step 1 All done." , flush=True )
        
    
    step=2
    if ( do_steps[ step-1 ] == True ):
        ###############################################
        # Step 2:
        #   Make perturbations
        ###############################################
        vars = ['U','V','T','Q','w_mpas','theta_mpas',]
        #vars = ['PINT'] 
        for var in vars:
            raw_var_file = f"{WrkDir}/{var}_dyamond.nc"
            bkg_var_file = f"{WrkDir}/{var}_dyamond_ne16pg3_mpasa3p75.{date_}.nc"
            prt_var_file = f"{WrkDir}/{var}_prt_dyamond.{date_}.nc"
    
            RawVar = xr.open_dataset( raw_var_file )
            BkgVar = xr.open_dataset( bkg_var_file )
            lon=RawVar.lon.values
            lat=RawVar.lat.values
    
            raw_var=RawVar[var].values
            bkg_var=BkgVar[var].values
    
            prt_var = raw_var - bkg_var
            
            print( f"Opened, extracted {raw_var_file}", flush=True )
            print( f"Opened, extracted {bkg_var_file}", flush=True )
            print( f"Calculated prt_{var}", flush=True )
    
            attrs = RawVar[var].attrs 
            dims  = RawVar[var].dims 
    
            print( f"{dims}", flush=True )
            
            if ('lev' in dims) and ('ilev' not in dims):
                coords = dict( 
                    time = ( ["time"], RawVar.time.values ),
                    lev  = ( ["lev"],  RawVar.lev.values),
                )
            elif ('lev' not in dims) and ('ilev' in dims):
                coords = dict( 
                    time = ( ["time"], RawVar.time.values ),
                    ilev = ( ["ilev"], RawVar.ilev.values),
                )
            elif ('lev' in dims) and ('ilev' in dims):
                coords = dict( 
                    time = ( ["time"], RawVar.time.values ),
                    lev  = ( ["lev"],  RawVar.lev.values),
                    ilev = ( ["ilev"], RawVar.ilev.values),
                )
            else:
                print( "croak" )
            
            
            Xout = xr.Dataset( coords=coords  )
            
            dims = ('ncol' )
            Dar = xr.DataArray( data=lon, 
                                dims=dims,)
            Xout[ 'lon' ]= Dar
            
            dims = ('ncol' )
            Dar = xr.DataArray( data=lat , 
                                dims=dims,)
            Xout[ 'lat' ]= Dar
            
            attrs = RawVar[var].attrs 
            dims  = RawVar[var].dims 
            
            Dar = xr.DataArray( data=prt_var , 
                                dims=dims,
                                attrs=attrs ,) 
                                    
            Xout[ f'{var}_prt' ]= Dar
    
            print(f"Writing {prt_var_file}", flush=True )
            
            Xout.to_netcdf( prt_var_file )
    
        print("Step 2 All done." , flush=True )
        
        
        
    step=3
    if ( do_steps[ step-1 ] == True ):
        ###############################################################
        # Step 3
        #    Calculate flux quadratics wp*(U,T, ... ) on mpasa3p75km
        ###############################################################
    
        var='w_mpas'
        prt_w_file = f"{WrkDir}/{var}_prt_dyamond.{date_}.nc"
    
        Wp=xr.open_dataset( prt_w_file )
        wp=Wp.w_mpas_prt.values
        wpdims = Wp.w_mpas_prt.dims
        lon=Wp.lon.values
        lat=Wp.lat.values
    
        flux_vars = ['U','V','T','Q','theta_mpas',]
        for var in flux_vars:
            prt_var_file = f"{WrkDir}/{var}_prt_dyamond.{date_}.nc"
            flx_var_file = f"{WrkDir}/{var}pwp_dyamond.{date_}.nc"
            Varp = xr.open_dataset( prt_var_file )
            varp = Varp[f"{var}_prt"].values
    
            nt,nz,ncol = np.shape( varp )
            varpi = np.zeros( (nt,nz+1,ncol) )
            varpi[:,0,:]=varp[:,0,:]
            varpi[:,nz,:]=varp[:,nz-1,:]
    
            print( f"Shape of {var} {nt,nz,ncol}" )
            
            for z in np.arange( start=1,stop=nz ):
                varpi[:,z,:] = 0.5*( varp[:,z-1,:] + varp[:,z,:] )
    
            
            #Validate z - interpolation (Debugging) 
            #z=0
            #print( f"- - - {varp[0,z,1_000_000]}", flush=True )
            #print( f"z={z} > {varpi[0,z,1_000_000]}", flush=True )
            #for z in np.arange( start=1,stop=nz+1 ):
            #    print( f"z={z-1} o o o {varp[0,z-1,1_000_000]}", flush=True )
            #    print( f"z={z} > {varpi[0,z,1_000_000]}", flush=True )
            
            print( f"finshed mid to edge interpol of {var} ", flush=True )
                            
            newvar=f"{var}pwp"
            new_attributes = {
                'long_name': 'flux {var}p x wp',}
            varpi_wp = (varpi * wp).astype("float32")
    
            print( f"Will write {newvar} with dims == {wpdims}", flush=True )
            print( f"shoae varpi_wp {np.shape(varpi_wp)}", flush=True )
    
            coords = dict( 
                time = ( ["time"], Wp.time.values ),
                ilev = ( ["ilev"], Wp.ilev.values), )
    
            Xout = xr.Dataset( coords=coords  )
            
            dims = ('ncol' )
            Dar = xr.DataArray( data=lon, 
                                dims=dims,)
            Xout[ 'lon' ]= Dar
            
            dims = ('ncol' )
            Dar = xr.DataArray( data=lat , 
                                dims=dims,)
            Xout[ 'lat' ]= Dar
                   
            Dar = xr.DataArray( data=varpi_wp , 
                                dims=wpdims,
                                attrs=new_attributes ,) 
                                    
            Xout[ f'{newvar}' ]= Dar
    
            print( f"wrting {newvar} with dims == {wpdims}", flush=True )
        
            Xout.to_netcdf( flx_var_file )
        
        print("Step 3 All done." , flush=True )
        
    
    step=4
    if ( do_steps[ step-1 ] == True ):
        ##############################################################
        # Step 4
        #   Coarse grain flux quadratics to ne16 and then project to fv1x1
        #############################################################
    
        flux_vars =['Upwp','Vpwp','Tpwp','Qpwp','theta_mpaspwp',]
        for var in flux_vars:
            infile   = f"{WrkDir}/{var}_dyamond.{date_}.nc"
            tmp      = f"{WrkDir}/{var}_dyamond_ne16pg3.nc"
            outfile= f"{WrkDir}/{var}_dyamond_ne16pg3_fv1x1.{date_}.nc"
            
            cmds = [
                # conservative remap mpasa3p75km -> ne16pg3
                "ncremap -m /glade/work/juliob/HiRes_ana_dev/Drivers/mpasa3p75_TO_ne16pg3_cnsrv.nc "
                f"{infile} {tmp}",
            
                # bilinear remap ne16pg3 -> fv1x1
                "ncremap -m /glade/work/juliob/HiRes_ana_dev/Drivers/ne16pg3_TO_fv1x1_bilin.nc "
                f"{tmp} {outfile}",
            ]
            
            for c in cmds:
                print("Running:", c)
                result = subprocess.run(c, shell=True)
            
                if result.returncode != 0:
                    raise RuntimeError(f"Command failed: {c}")
            
        print("Step 4 All done." , flush=True )
        
    step=5
    if ( do_steps[ step-1 ] == True ):
        ###############################################
        # Step 5
        #   Make other dynamical fileds on fv1x1 grid
        ###############################################
    
        #vars = ['U','V','T','Q','theta_mpas','rho_mpas','PINT',]
        vars = ['PRECC','PRECL',]
        for var in vars:
            infile   = f"{WrkDir}/{var}_dyamond.nc"
            outfile= f"{WrkDir}/{var}_dyamond_fv1x1.{date_}.nc"
            
            cmds = [
                # conservative remap mpasa3p75km -> fv1x1
                "ncremap -m /glade/work/juliob/HiRes_ana_dev/Drivers/mpasa3p75_TO_fv1x1_cnsrv.nc "
                f"{infile} {outfile}",
            
            ]
            
            for c in cmds:
                print("Running:", c)
                result = subprocess.run(c, shell=True)
            
                if result.returncode != 0:
                    raise RuntimeError(f"Command failed: {c}")
            
        print("Step 5 All done." , flush=True )
        

    step=6
    if ( do_steps[ step-1 ] == True ):
        ##############################################################
        # Step 6
        #   Make some dyn fields on a UHR (lat-lon) regional grid
        ##############################################################
        #UHRtag="UHR_SAndesAP"
        #UHRtag="UHR_SO-Indian"

        WrkDirUHR = f"/glade/campaign/cgd/amp/juliob/mpasa3p75km/{case}/{UHRtag}/{date_}"
        os.makedirs( WrkDirUHR , exist_ok=True )

        """
        # extract PHIS
        var='PHIS'
        infile = "/glade/campaign/cgd/amp/aherring/mpas-uniform/mpasa3p75/topo/mpasa3p75_gmted2010_modis_bedmachine_nc3000_Laplace0004_noleak_20241006.nc"
        outfile= f"{WrkDir}/{var}_dyamond.nc"

        cmds = [
            # extract variable
            f"ncks -O -v {var},lat,lon {infile} {outfile}",
               ]
        """
        """
        for c in cmds:
            print("Running:", c)
            result = subprocess.run(c, shell=True)
        
            if result.returncode != 0:
                raise RuntimeError(f"Command failed: {c}")
        """

        vars =['PHIS','zgrid',]
        for var in vars:
            infile   = f"{TimeInvDir}/{var}_dyamond.nc"
            outfile= f"{TimeInvDir}/{var}_dyamond_{UHRtag}.nc"

            nxoo, nyoo = 512, 512
            bytes_per_real8 = 8
            min_payload = nxoo * nyoo * bytes_per_real8
            min_size = int(1.5 * min_payload)   # add safety factor
            
            outpath = Path(outfile)
            outsize = outpath.stat().st_size if outpath.is_file() else 0

            if ( outsize < min_size ):
                print( f"doing {outfile} " , flush=True ) 
                cmds = [
                    # conservative remap mpasa3p75km -> UHR
                    f"ncremap -m /glade/work/juliob/HiRes_ana_dev/Drivers/mpasa3p75_TO_{UHRtag}_bilin.nc "
                    f"{infile} {outfile}",
                            ]
                for c in cmds:
                    print("Running:", c)
                    result = subprocess.run(c, shell=True)
                
                    if result.returncode != 0:
                        raise RuntimeError(f"Command failed: {c}")
            else:
                print( f"{outfile} EXITS and seems big enuff " , flush=True ) 
            
        vars =['U','V','T','Q','theta_mpas','w_mpas','rho_mpas',]
        for var in vars:
            infile   = f"{WrkDir}/{var}_dyamond.nc"
            outfile= f"{WrkDirUHR}/{var}_dyamond_{UHRtag}.{date_}.nc"
            
            cmds = [
                # conservative remap mpasa3p75km -> UHR
                f"ncremap -m /glade/work/juliob/HiRes_ana_dev/Drivers/mpasa3p75_TO_{UHRtag}_bilin.nc "
                f"{infile} {outfile}",
                        ]
            
            for c in cmds:
                print("Running:", c)
                result = subprocess.run(c, shell=True)
            
                if result.returncode != 0:
                    raise RuntimeError(f"Command failed: {c}")
            
        vars =['U_prt','V_prt','theta_mpas_prt','w_mpas_prt',]
        for var in vars:
            infile   = f"{WrkDir}/{var}_dyamond.{date_}.nc"
            outfile= f"{WrkDirUHR}/{var}_dyamond_{UHRtag}.{date_}.nc"
            
            cmds = [
                # conservative remap mpasa3p75km -> UHR
                f"ncremap -m /glade/work/juliob/HiRes_ana_dev/Drivers/mpasa3p75_TO_{UHRtag}_bilin.nc "
                f"{infile} {outfile}",
                        ]
            
            for c in cmds:
                print("Running:", c)
                result = subprocess.run(c, shell=True)
            
                if result.returncode != 0:
                    raise RuntimeError(f"Command failed: {c}")
            
        print("Step 6 All done." , flush=True )
    
    step=7
    if ( do_steps[ step-1 ] == True ):
        ###############################################################
        # Step 7
        #    Calculate 2nd set quadratics Tp2, wp2 on mpasa3p75km
        ###############################################################
    
        flux_vars = ['T','theta_mpas','w_mpas',]
        quad_vars = ['T','theta','w',]
        
        for var,qvar in zip(flux_vars, quad_vars):
            prt_var_file = f"{WrkDir}/{var}_prt_dyamond.{date_}.nc"
            qua_var_file = f"{WrkDir}/{qvar}p2_dyamond.{date_}.nc"
            Varp = xr.open_dataset( prt_var_file )
            varp = Varp[f"{var}_prt"].values
            vdims = Varp[f"{var}_prt"].dims
            lon=Varp.lon.values
            lat=Varp.lat.values
            varp2 = varp**2
            
            newvar=f"{qvar}p2"
            new_attributes = {
                'long_name': f"{var}p **2",}
    
            print( f"Quad of {var} in {prt_var_file}", flush=True )
            print( f"Will write {newvar} with dims == {vdims}", flush=True )
    
            if ('lev' in vdims):
                coords = dict( 
                    time = ( ["time"], Varp.time.values ),
                    lev = ( ["lev"], Varp.lev.values), )
            elif ('ilev' in vdims):
                coords = dict( 
                    time = ( ["time"], Varp.time.values ),
                    ilev = ( ["ilev"], Varp.ilev.values), )
    
            Xout = xr.Dataset( coords=coords  )
            
            dims = ('ncol' )
            Dar = xr.DataArray( data=lon, 
                                dims=dims,)
            Xout[ 'lon' ]= Dar
            
            dims = ('ncol' )
            Dar = xr.DataArray( data=lat , 
                                dims=dims,)
            Xout[ 'lat' ]= Dar
                   
            Dar = xr.DataArray( data=varp2 , 
                                dims=vdims,
                                attrs=new_attributes ,) 
                                    
            Xout[ f'{newvar}' ]= Dar
    
            print( f"wrting {newvar} with dims == {vdims}", flush=True )
        
            Xout.to_netcdf( qua_var_file )
        
        print("Step 7 All done." , flush=True )
        
    
    step=8
    if ( do_steps[ step-1 ] == True ):
        ##############################################################
        # Step 8
        #   Coarse grain 2nd set quadratics to ne16 and then project to fv1x1
        #############################################################
    
        quad_vars =['Tp2','wp2','thetap2',]
        for var in quad_vars:
            infile   = f"{WrkDir}/{var}_dyamond.{date_}.nc"
            tmp      = f"{WrkDir}/{var}_dyamond_ne16pg3.nc"
            outfile= f"{WrkDir}/{var}_dyamond_ne16pg3_fv1x1.{date_}.nc"
            
            cmds = [
                # conservative remap mpasa3p75km -> ne16pg3
                "ncremap -m /glade/work/juliob/HiRes_ana_dev/Drivers/mpasa3p75_TO_ne16pg3_cnsrv.nc "
                f"{infile} {tmp}",
            
                # bilinear remap ne16pg3 -> fv1x1
                "ncremap -m /glade/work/juliob/HiRes_ana_dev/Drivers/ne16pg3_TO_fv1x1_bilin.nc "
                f"{tmp} {outfile}",
            ]
            
            for c in cmds:
                print("Running:", c)
                result = subprocess.run(c, shell=True)
            
                if result.returncode != 0:
                    raise RuntimeError(f"Command failed: {c}")
            
        print("Step 8 All done." , flush=True )
        
    step=9
    if ( do_steps[ step-1 ] == True ):
        ##############################################################
        # Step 9:
        #   Combine all fv1x1 files
        #############################################################
        outfile = f"{WrkDir}/DynVars_dyamond_fv1x1.{date_}.nc"
        files = [
            f"{WrkDir}/Q_dyamond_fv1x1.{date_}.nc",
            f"{WrkDir}/Qpwp_dyamond_ne16pg3_fv1x1.{date_}.nc",
            f"{WrkDir}/rho_mpas_dyamond_fv1x1.{date_}.nc",
            f"{WrkDir}/T_dyamond_fv1x1.{date_}.nc",
            f"{WrkDir}/theta_mpas_dyamond_fv1x1.{date_}.nc",
            f"{WrkDir}/theta_mpaspwp_dyamond_ne16pg3_fv1x1.{date_}.nc",
            f"{WrkDir}/thetap2_dyamond_ne16pg3_fv1x1.{date_}.nc",
            f"{WrkDir}/Tp2_dyamond_ne16pg3_fv1x1.{date_}.nc",
            f"{WrkDir}/Tpwp_dyamond_ne16pg3_fv1x1.{date_}.nc",
            f"{WrkDir}/U_dyamond_fv1x1.{date_}.nc",
            f"{WrkDir}/Upwp_dyamond_ne16pg3_fv1x1.{date_}.nc",
            f"{WrkDir}/V_dyamond_fv1x1.{date_}.nc",
            f"{WrkDir}/Vpwp_dyamond_ne16pg3_fv1x1.{date_}.nc",
            f"{WrkDir}/wp2_dyamond_ne16pg3_fv1x1.{date_}.nc",
            f"{WrkDir}/PINT_dyamond_fv1x1.{date_}.nc",
            f"{TimeInvDir}/zgrid_dyamond_fv1x1.nc",
        ]
        X=xr.open_dataset( files[0] )
        for f in files[1:]:
            var = os.path.basename(f).split("_dyamond_")[0]
            print( f"{var} in {f} ",flush=True )
            X2 = xr.open_dataset( f )
            #X[var] = X2[var]
            # More convoluted strategy needed fro cam77 since probably
            # reconfiguring {w,theta,rho} to {...}_mpas has effed up time 
            # coord ... leading to 'alignment' catastrophe with simple 
            # line above (per ChatGPT)
            X[var] = xr.DataArray(
                X2[var].values,
                dims=X2[var].dims
            )

        print( f"Upwp value range :, {np.min(X.Upwp.values)} , {np.max(X.Upwp.values)}", flush=True )
        print( list(X.variables) , flush=True )
        print( f"Vpwp value range :, {np.min(X.Vpwp.values)} , {np.max(X.Vpwp.values)}", flush=True )
        print( list(X.variables) , flush=True )
        print( f"Tpwp value range :, {np.min(X.Tpwp.values)} , {np.max(X.Tpwp.values)}", flush=True )
        print( list(X.variables) , flush=True )

        X.to_netcdf( outfile )
        
        print("Step 9 All done." , flush=True )

    step=10
    if ( do_steps[ step-1 ] == True ):
        ##############################################################
        # Step 10:
        #   Regrid time-invariant grid files like zgrid to fv1x1
        #############################################################

        vars =['PHIS','zgrid',]
        for var in vars:
            infile   = f"{TimeInvDir}/{var}_dyamond.nc"
            outfile= f"{TimeInvDir}/{var}_dyamond_fv1x1.nc"
            
            nxoo, nyoo = 512, 512
            bytes_per_real8 = 8
            min_payload = nxoo * nyoo * bytes_per_real8
            min_size = int(1.5 * min_payload)   # add safety factor
            
            outpath = Path(outfile)
            outsize = outpath.stat().st_size if outpath.is_file() else 0

            if ( outsize < min_size ):
                print( f" {outfile} being made ... ", flush=True )
                cmds = [
                    # conservative remap mpasa3p75km -> fv1x1
                    "ncremap -m /glade/work/juliob/HiRes_ana_dev/Drivers/mpasa3p75_TO_fv1x1_cnsrv.nc "
                    f"{infile} {outfile}",
                
                ]
                
                for c in cmds:
                    print("Running:", c)
                    result = subprocess.run(c, shell=True)
                
                    if result.returncode != 0:
                        raise RuntimeError(f"Command failed: {c}")


            else:
                print( f" {outfile} ALREADY EXISTS ... ", flush=True )
               
        print("Step 10 All done." , flush=True )

def fix_cam77_run( tmp1x, tmp1, varx, var ):
    
    X=xr.open_dataset( tmp1x )

    var_dims = X[varx].dims

    if ( ( 'lev' in var_dims ) and ( 'ilev' not in var_dims ) ):
        coords = dict( 
            ncol = ( ["ncol"],X.ncol.values ),
            lev  = ( ["lev"],X.lev.values ),
        )
    elif ( ( 'lev' not in var_dims ) and ( 'ilev' in var_dims ) ):
        coords = dict( 
            ncol = ( ["ncol"],X.ncol.values ),
            ilev = ( ["ilev"],X.ilev.values ),
        )
    elif ( ('lev' in var_dims ) and ( 'ilev' in var_dims ) ):
        coords = dict( 
            ncol = ( ["ncol"],X.ncol.values ),
            lev  = ( ["lev"],X.lev.values ),
            ilev = ( ["ilev"],X.ilev.values ),
        )
    
    Xo = xr.Dataset( coords=coords  )
    Xo.coords['time']=X.coords['time']
    
    
    lon=X.lon.values
    lat=X.lat.values
    var_data = X[varx].values    
    lev_ = var_dims[1]
    
    Dar = xr.DataArray( data=lat , 
                        dims=('ncol',),
                        attrs=dict( long_name='latitude',) ,) 
    
    
    Xo['lat']=Dar
    
    Dar = xr.DataArray( data=lon , 
                        dims=('ncol',),
                        attrs=dict( long_name='longitude',) ,) 
    
    
    Xo['lon']=Dar
    
    Dar = xr.DataArray( data=var_data , 
                        dims=( 'time', lev_ ,'ncol',),
                        attrs=dict( long_name='altitude of grid edges (m)',) ,) 
    
    
    Xo[var]=Dar
    
    Xo.to_netcdf( tmp1 )

if __name__ == "__main__":
    drive()


