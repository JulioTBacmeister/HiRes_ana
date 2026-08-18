# The usual
from datetime import date
import numpy as np
import xarray as xr
import event_io as eio
import analysis_utils as auti

#================================================================
def reset_hyperparameters():
        
    #====================================
    # Default Hyperparameters (from Will)
    #====================================
    # --- train/test split
    train_interval   = (48, 248)        # same split as the RF
    test_interval    = (0, 28)
    # --- architecture ---
    hidden_dims=           (256, 256, 256, 128)
    dropout=              0.1
    # --- training ---
    batch_size=             512
    lr=                     1e-3
    weight_decay=           1e-5
    patience=               100
    max_epochs=             500
    # --- lr schedule ---
    lr_patience=           3
    lr_factor=              0.5
    min_lr=                 1e-6
    # --- loss ---
    loss_power=            5
    # --- predictor transforms ---
    log_predictor_patterns= ['tilt', 'U_wv_src_mm', 'abs_zeta']
    
    
    #======================================
    # My predictor "hyperparameters"
    #======================================
    predictorSet='None given'  
    trange,yrange,xrange = None,None,None
    trange_targ,yrange_targ,xrange_targ = None,None,None
    z_targ,targ_scaling=None,None
    use_predictors=None
    use_MM_winds=False
    time_average=False
    
    
    MyHyperparameters = {
        # --- predictor set ---
        'predictorSet':           predictorSet,
        'trange':                 trange,
        'yrange':                 yrange,
        'xrange':                 xrange,
        'use_predictors':         use_predictors,
        'use_MM_winds':           use_MM_winds,
        'time_average':           time_average,
        # --- target settings ---
        'z_targ':                 z_targ,
        'targ_scaling':           targ_scaling,
        'trange_targ':            trange_targ,
        'yrange_targ':            yrange_targ,
        'xrange_targ':            xrange_targ,
        # --- architecture ---
        'hidden_dims':            hidden_dims,
        'dropout':                dropout,
        # --- training ---
        'batch_size':             batch_size,
        'lr':                     lr,
        'weight_decay':           weight_decay,
        'patience':               patience,
        'max_epochs':             max_epochs,
        # --- lr schedule ---
        'lr_patience':            lr_patience,
        'lr_factor':              lr_factor,
        'min_lr':                 min_lr,
        # --- loss ---
        'loss_power':             loss_power,
        # --- predictor transforms ---
        'log_predictor_patterns': log_predictor_patterns,
    }

    return MyHyperparameters

#========================================================
def build_predictor_set(Eco, zlev, MyHyperparameters):
    """
    Single entry point for all predictor sets.
    Reads everything it needs from MyHyperparameters.
    Returns predictors, predictor_names, key_z, short_desc, yv
    """
    predictorSet   = MyHyperparameters['predictorSet']
    trange         = MyHyperparameters.get('trange',        None)
    yrange         = MyHyperparameters.get('yrange',        None)
    xrange         = MyHyperparameters.get('xrange',        None)
    use_MM_winds   = MyHyperparameters.get('use_MM_winds',  False)
    time_average   = MyHyperparameters.get('time_average',  False)
    targ_scaling   = MyHyperparameters.get('targ_scaling',  1.0)
    z_targ         = MyHyperparameters.get('z_targ',        None)
    trange_targ    = MyHyperparameters.get('trange_targ',   None)
    yrange_targ    = MyHyperparameters.get('yrange_targ',   None)
    xrange_targ    = MyHyperparameters.get('xrange_targ',   None)
    use_predictors = MyHyperparameters.get('use_predictors',None)
    z_launch       = MyHyperparameters.get('z_launch',      None)
    z_steer        = MyHyperparameters.get('z_steer',       None)

    # for dynamic steer-launch
    zl_dyn         = None
    zs_dyn         = None
    
    if predictorSet == 'original_default':
        predictors, predictor_names, use_predictors, key_z, short_desc = \
            set_A_MM_genl(Eco=Eco, zlev=zlev,
                          trange=trange, yrange=yrange, xrange=xrange,
                          use_predictors=use_predictors,
                          use_MM_winds=False)

    elif predictorSet == 'tilt_zeta_stab_MM_z10':
        predictors, predictor_names, use_predictors, key_z, short_desc = \
            set_A_MM_genl(Eco=Eco, zlev=zlev,
                          trange=trange, yrange=yrange,
                          use_predictors=use_predictors,
                          use_MM_winds=True)

    elif predictorSet == 'GWP-like':
        predictors, predictor_names, use_predictors, key_z, short_desc = \
            GWP_like_set(Eco=Eco, zlev=zlev,
                         trange=trange, yrange=yrange,
                         use_predictors=use_predictors,
                         time_average=time_average)

    elif predictorSet == 'MovMtn-like':
        predictors, predictor_names, use_predictors, key_z, short_desc = \
            MovMtn_like_set(Eco=Eco, zlev=zlev,z_steer=z_steer,z_launch=z_launch,
                         trange=trange, yrange=yrange,
                         use_predictors=use_predictors,
                         time_average=time_average)

    elif predictorSet == 'Probe-steer-launch':
        predictors,predictor_names,use_predictors,key_z,short_desc = \
            probe_steer_launch(Eco=Eco,zlev=zlev, zlev_steer=z_steer, 
                               zlev_launch=z_launch,
                               trange=trange, yrange=yrange,
                               use_predictors=use_predictors )

    elif predictorSet == 'Dynamic-steer-launch':
        predictors,predictor_names,use_predictors,key_z,short_desc,zs_dyn,zl_dyn = \
            dynamic_steer_launch(Eco=Eco,zlev=zlev, 
                               trange=trange, yrange=yrange,xrange=xrange,
                               use_predictors=use_predictors, verbose=True )
    else:
        raise ValueError(f"Unknown predictorSet: '{predictorSet}'. "
                         f"Check MyHyperparameters['predictorSet'].")

    # --- target ---
    yv = build_target( Eco, MyHyperparameters , zl_dyn=zl_dyn)
    #yv = targ_scaling * np.mean(Eco.epwp_4D[:, -1, z_targ, :, :], axis=(2, 1))
    #yv = targ_scaling * np.mean(Eco.epwp_4D[:, -1, z_targ, 3:-3, 3:-3], axis=(2, 1))

    # --- summary ---
    print(f"{'='*60}")
    print(f"Predictor set    : {predictorSet}")
    print(f"Fields           : {use_predictors}")
    print(f"n_predictors     : {len(predictors)}")
    #print(f"key_z levels (m) : {[int(zlev[z]) for z in key_z]}")
    print(f"trange           : {trange}")
    print(f"yrange           : {yrange}")
    print(f"xrange           : {xrange}")
    print(f"time_average     : {time_average}")
    print(f"use_MM_winds     : {use_MM_winds}")
    print(f"z_targ (m)       : {zlev[z_targ]:.0f}")
    print(f"targ_scaling     : {targ_scaling}")
    print(f"Target           : mean={yv.mean():.4f}  std={yv.std():.4f}")
    print(f"Case             : {Eco.case}")
    print(f"lon x lat        : {Eco.lon_range} x {Eco.lat_range}")
    print(f"exclude orog     : {Eco.exclude_orography}")
    print(f"peak footprint   : {Eco.peak_footprint}")
    print(f"{'='*60}")
    
    target_ = f"Target: {Eco.fld} at {zlev[z_targ]/1000.:.0f}km"
    short_desc=f"{target_}: {short_desc}"

    return predictors, predictor_names, use_predictors, key_z, short_desc, yv

###########################################
def build_target( Eco, MyHyperparameters, zl_dyn=None):

    print( "In build_target function " )
    targ_scaling   = MyHyperparameters.get('targ_scaling',  1.0)
    z_targ         = MyHyperparameters.get('z_targ',        None)
    trange_targ    = MyHyperparameters.get('trange_targ',   None)
    yrange_targ    = MyHyperparameters.get('yrange_targ',   None)
    xrange_targ    = MyHyperparameters.get('xrange_targ',   None)

    if zl_dyn is None:
        # --- original default: full horz avg across cube at t=0 (-1) ---
        yv = targ_scaling * np.mean(Eco.epwp_4D[:, -1, z_targ, :, :], axis=(2, 1))
        # --- Restricted 0:+/-1 avg across cube at t=0 (-1) ---
        #yv = targ_scaling * np.mean(Eco.epwp_4D[:, -1, z_targ, 3:-3, 3:-3], axis=(2, 1))
    else:
        print(f"trying dynamic launch level: shape = {np.shape(zl_dyn)}")
        nv,nt_v,nz_v,ny_v,nx_v = np.shape( Eco.epwp_4D )
        yv = np.zeros( (nv) )
        for v in np.arange( nv ):
            yv[v] = targ_scaling * np.mean(Eco.epwp_4D[v, -1, zl_dyn[v,-1], :, :], axis=(0,1) )

    return yv

#######################################################
def set_A(Eco=None,zlev=None):
    
    nv,nt_v,nz_v,ny_v,nx_v = np.shape( Eco.zeta_4D )

    z0=np.argmin( np.abs( zlev-0.))
    z0p5=np.argmin( np.abs( zlev-500))
    z1=np.argmin( np.abs( zlev-1000.))
    z3=np.argmin( np.abs( zlev-3000.))
    z5=np.argmin( np.abs( zlev-5000.))
    z6=np.argmin( np.abs( zlev-6000.))
    z7=np.argmin( np.abs( zlev-7000.))
    z10=np.argmin( np.abs( zlev-10000.))
    z11=np.argmin( np.abs( zlev-11000.))
    z12=np.argmin( np.abs( zlev-12000.))
    z15=np.argmin( np.abs( zlev-15000.))
    
    predictors=[]
    predictor_names=[]
    
    pred_scaling=1.
    targ_scaling=1.
    # use_predictors=['u_4D','v_4D','tilt_4D']
    use_predictors = ['u_4D', 'v_4D', 'tilt_4D', 'zeta_4D']
    #use_predictors = ['u_4D', 'v_4D', 'fgf_4D', 'zeta_4D']
    
    #key_z = [z0p5, z1, z3, z5, z6, z7, z10, z11, z12, z15]
    key_z = [z0p5, z1, z3, z5, z6, z7, z10]
    for t in np.arange(nt_v):
        for z in key_z:
            for prd in use_predictors:
                predictors.append(np.mean(Eco[prd][:,t,z,0:6,:], axis=(1,2)))
                predictor_names.append(f"{prd} t={t-nt_v+1}, z={zlev[z]:.0f}")

    print( f"Created predictor set=A" )

    return predictors,predictor_names,use_predictors,key_z

######################################################################
def probe_steer_launch(Eco=None,zlev=None,use_predictors=None,
                 zlev_steer=None,zlev_launch=None,
                 trange=None,yrange=None,xrange=None,
                 verbose=False):
    """
    Only Eco and zlev are mandatory.
    """
    
    if (zlev_launch is None):
        zlev_launch = np.arange( 0, 13 ) * 1000.
    if (zlev_steer is None):
        zlev_steer = np.arange( 0, 13 ) * 1000.

    
    z_launch = []
    for zl in zlev_launch:
        z_launch.append( np.argmin( np.abs( zlev-zl)) )

    z_steer = []
    for zl in zlev_launch:
        z_steer.append( np.argmin( np.abs( zlev-zl)) )

    key_z = {"steer":zlev_steer,"launch":zlev_launch}

    nv,nt_v,nz_v,ny_v,nx_v = np.shape( Eco.zeta_4D )
    
    predictors=[]
    predictor_names=[]


    if trange is None:
        start_t, stop_t =  0 , nt_v # No Mem=> nt_v-1, nt_v 
    else:
        start_t, stop_t = trange
    if yrange is None:
        j0,j1= 0,6  
    else:
        j0,j1= yrange         
    if xrange is None:
        i0,i1= 0,nx_v  # -1 here is not eqv  - Note, Claude says use 'None'
    else:
        i0,i1= xrange 
    
    pred_scaling=1.
    targ_scaling=1.
    # use_predictors=['u_4D','v_4D','tilt_4D']

    if use_predictors is None:
        use_predictors = ['abs_zeta_4D']


    for zs in z_steer:
        for zl in z_launch:
            if zs > zl:
                #print( f"{zlev[zs]:.0f} -- {zlev[zl]:.0f} : legit pair")
                for t in np.arange(start=start_t,stop=stop_t ):
                    u_steer=np.mean(Eco['u_4D'][:,t,zs,j0:j1,i0:i1], axis=(1,2))
                    v_steer=np.mean(Eco['v_4D'][:,t,zs,j0:j1,i0:i1], axis=(1,2))
                    u_lnch= np.mean(Eco['u_4D'][:,t,zl,j0:j1,i0:i1], axis=(1,2)) - u_steer
                    v_lnch= np.mean(Eco['v_4D'][:,t,zl,j0:j1,i0:i1], axis=(1,2)) - v_steer
                    usrc_mm = np.sqrt( u_lnch**2 + v_lnch**2 )
                    predictors.append(usrc_mm)
                    this_name = f"U_wv_src_mm t={t-nt_v+1}, zsteer:{zlev[zs]:.0f} zlnch:{zlev[zl]:.0f}"
                    predictor_names.append( this_name )
                    if verbose:
                        print( this_name )
    
    for zs in z_steer:
        for zl in z_launch:
            if zs > zl:
                #print( f"{zlev[zs]:.0f} -- {zlev[zl]:.0f} : legit pair")
                for pred_ in use_predictors:
                    for t in np.arange(start=start_t,stop=stop_t ):
                        if pred_ == 'abs_zeta_4D':
                            pred_arr = np.mean( np.abs(Eco['zeta_4D'])[:,t,zl:zs,j0:j1,i0:i1], axis=(1,2,3))
                            this_name = f"avg({pred_}) t={t-nt_v+1}, zsteer:{zlev[zs]:.0f} zlnch:{zlev[zl]:.0f}"
                        else:
                            pred_arr = np.mean( Eco[pred_][:,t,zl:zs,j0:j1,i0:i1], axis=(1,2,3))
                            this_name = f"avg({pred_}) t={t-nt_v+1}, zsteer:{zlev[zs]:.0f} zlnch:{zlev[zl]:.0f}"
                            
                        predictors.append(pred_arr)
                        predictor_names.append( this_name )
                        if verbose:
                            print( this_name )

    
    print( f"Created predictor set to explore steer and launch " )


    if i1 > 0:
        i1p=i1-1
    else:
        i1p=nx_v+i1-1
    if j1 > 0:
        j1p=j1-1
    else:
        j1p=ny_v+j1-1
        
    filedesc = eio.make_base_filename(Eco)
    short_desc=f"Window size:( {nt_v},{ny_v},{nx_v} ), Sub-window: T({start_t},{stop_t-1}),Y({j0},{j1p}), X({i0},{i1p})"
    short_desc=short_desc + '\n' + filedesc
    
    n = 4
    predlist = '\n'.join(',  '.join(predictor_names[i:i+n]) for i in range(0, len(predictor_names), n))
    short_desc=short_desc + '\n' + '\n' + predlist

    return predictors,predictor_names,use_predictors,key_z,short_desc

######################################################################
def dynamic_steer_launch(Eco=None,zlev=None,use_predictors=None,
                 trange=None,yrange=None,xrange=None,t_avg_flux_pred=False,
                 verbose=False,debug=False):
    """
    Only Eco and zlev are mandatory.
    """

    if 'abs_zeta_4D' not in Eco:
        Eco['abs_zeta_4D'] = np.abs( Eco.zeta_4D)
        print( f"aded abs_zeta_4D ot Eco ")
        
    
    plev=100_000. * np.exp( -zlev / 7000. )

    nv,nt_v,nz_v,ny_v,nx_v = np.shape( Eco.zeta_4D )
    
    pmid = np.tile( plev , (nv,1) )
    
    predictors=[]
    predictor_names=[]


    if trange is None:
        start_t, stop_t =  0 , nt_v # No Mem=> nt_v-1, nt_v 
    else:
        start_t, stop_t = trange
    if yrange is None:
        j0,j1= 0,ny_v 
    else:
        j0,j1= yrange         
    if xrange is None:
        i0,i1= 0,nx_v  # -1 here is not eqv  - Note, Claude says use 'None'
    else:
        i0,i1= xrange 
    
    pred_scaling=1.
    targ_scaling=1.

    # First predictor is used to find z_steer and z_launch.
    # Subsequent predictors are used for initial flux ampl.
    
    if use_predictors is None:
        use_predictors = ['vmag_4D','abs_zeta_4D','tilt_4D']

    pred0_n = use_predictors[0]

    if pred0_n == 'vmag_4D':
        pred0 = np.sqrt(Eco['u_4D']**2 + Eco['v_4D']**2 )
    elif pred0_n == 'abs_zeta_4D':
        pred0 = np.abs(Eco['zeta_4D'])
    else:
        pred0 = Eco[pred0_n]

    key_z = f"Dynamic steering and launch based on {pred0_n}"

    print( key_z )

    
    zs = np.zeros( (nv,nt_v) , dtype=int )
    zl = np.zeros( (nv,nt_v) , dtype=int )

    plev_s = np.zeros( (nv,nt_v) )
    plev_l = np.zeros( (nv,nt_v) )
    zoop   = np.zeros( (nv,nt_v,nz_v) )

    u_steer = np.zeros( (nv) )
    v_steer = np.zeros( (nv) )
    u_lnch  = np.zeros( (nv) )
    v_lnch  = np.zeros( (nv) )
    usrc_mm = np.zeros( (nv) )
    pred_arr = np.zeros( (nv) )


    skip_ZL = ['thpwp_4D','stab_4D','precl_4D']
    skip_ZSxZL = ['thpwp_4D','stab_4D','precl_4D']

    skip_ZS = ['']
    skip_0xZS = ['precl_4D',]
    
    for t in np.arange(start=start_t,stop=stop_t ):
        zoop[:,t,:] = np.mean( pred0[:,t,:,j0:j1,i0:i1], axis=(2,3) )   # dims - (nv,nz_v)
        steering_level, launch_level, c_steer, c_launch = auti.vorticity_centroid_levels(vorticity = zoop[:,t,:] , coord=pmid , bound_surface=85_000., bound_top=5_000. )
        zs[:,t] = steering_level 
        zl[:,t] = launch_level 
        plev_s[:,t] = c_steer
        plev_l[:,t] = c_launch


    for t in np.arange(start=start_t,stop=stop_t ):
        for v in np.arange( nv ):
            u_steer[v]  = np.mean(Eco['u_4D'][v,t, zs[v,t] ,j0:j1,i0:i1], axis=(0,1))
            v_steer[v]  = np.mean(Eco['v_4D'][v,t, zs[v,t] ,j0:j1,i0:i1], axis=(0,1))
            u_lnch[v]   = np.mean(Eco['u_4D'][v,t, zl[v,t] ,j0:j1,i0:i1], axis=(0,1)) - u_steer[v]
            v_lnch[v]   = np.mean(Eco['v_4D'][v,t, zl[v,t] ,j0:j1,i0:i1], axis=(0,1)) - v_steer[v]
            usrc_mm[v]  = np.sqrt( u_lnch[v]**2 + v_lnch[v]**2 )
        predictors.append(usrc_mm.copy())
        this_name = f"U_wv_src_mm t={t-nt_v+1}, dynamic"
        predictor_names.append( this_name )
        if verbose:
            print(f" {this_name}  , shape = {np.shape(usrc_mm)}  ,minmax={np.min(usrc_mm.copy())} - {np.max(usrc_mm.copy())}")

    for pred_ in use_predictors[1:]:
        if (t_avg_flux_pred==False):
            for t in np.arange(start=start_t,stop=stop_t ):
                pred_arr = np.zeros( (nv) )
                for v in np.arange( nv ):
                    zs0,zl0 = zs[v,t] , zl[v,t] 
                    #zs0,zl0 = zl[v,t]+2, zl[v,t]-2 
                    pred_arr[v] = np.mean( Eco[pred_][v,t,zl0:zs0+1,j0:j1,i0:i1], axis=(0,1,2))
                if (pred_ not in skip_ZSxZL ):
                    # Don't use avg thpwp at ZS-ZL
                    predictors.append(pred_arr.copy() )
                    this_name = f"avg({pred_}) t={t-nt_v+1}, dynamic z (ZS-ZL avg)"
                    predictor_names.append( this_name )
                    if verbose:
                        print(f" {this_name}  , shape = {np.shape(pred_arr)} ,minmax={np.min(pred_arr.copy())} - {np.max(pred_arr.copy())}")
    
                pred_arr = np.zeros( (nv) )
                for v in np.arange( nv ):
                    zs0,zl0 = zl[v,t], zl[v,t]
                    pred_arr[v] = np.mean( Eco[pred_][v,t,zl0:zs0+1,j0:j1,i0:i1], axis=(0,1,2))
                if (pred_ not in skip_ZL ):
                    # Don't use thpwp at ZL
                    predictors.append(pred_arr.copy() )
                    this_name = f"avg({pred_}) t={t-nt_v+1}, dynamic z (ZL)"
                    predictor_names.append( this_name )
                    if verbose:
                        print(f" {this_name}  , shape = {np.shape(pred_arr)} ,minmax={np.min(pred_arr.copy())} - {np.max(pred_arr.copy())}")
    
                pred_arr = np.zeros( (nv) )
                for v in np.arange( nv ):
                    zs0,zl0 = zs[v,t], zs[v,t]
                    pred_arr[v] = np.mean( Eco[pred_][v,t,zl0:zs0+1,j0:j1,i0:i1], axis=(0,1,2))
                if (pred_ not in skip_ZS ):
                    predictors.append(pred_arr.copy() )
                    this_name = f"avg({pred_}) t={t-nt_v+1}, dynamic z (ZS)"
                    predictor_names.append( this_name )
                    if verbose:
                        print(f" {this_name}  , shape = {np.shape(pred_arr)} ,minmax={np.min(pred_arr.copy())} - {np.max(pred_arr.copy())}")
    
                pred_arr = np.zeros( (nv) )
                for v in np.arange( nv ):
                    zs0,zl0 = nz_v-1, zs[v,t]
                    pred_arr[v] = np.mean( Eco[pred_][v,t,zl0:zs0+1,j0:j1,i0:i1], axis=(0,1,2))
                if (pred_ not in skip_0xZS ):
                    predictors.append(pred_arr.copy() )
                    this_name = f"avg({pred_}) t={t-nt_v+1}, dynamic z (0-ZS avg)"
                    predictor_names.append( this_name )
                    if verbose:
                        print(f" {this_name}  , shape = {np.shape(pred_arr)} ,minmax={np.min(pred_arr.copy())} - {np.max(pred_arr.copy())}")
        else:
            pred_arr = np.zeros( (nv) )
            for v in np.arange( nv ):
                zs0,zl0 = zs[v,t] , zl[v,t] 
                #zs0,zl0 = zl[v,t]+2, zl[v,t]-2 
                pred_arr[v] = np.mean( Eco[pred_][v,:,zl0:zs0+1,j0:j1,i0:i1], axis=(0,1,2,3))
            predictors.append(pred_arr.copy() )
            this_name = f"avg({pred_}) t-AVG, dynamic z (ZS-ZL avg)"
            predictor_names.append( this_name )
            if verbose:
                print(f" {this_name}  , shape = {np.shape(pred_arr)} ,minmax={np.min(pred_arr.copy())} - {np.max(pred_arr.copy())}")

            pred_arr = np.zeros( (nv) )
            for v in np.arange( nv ):
                zs0,zl0 = zl[v,t], zl[v,t]
                pred_arr[v] = np.mean( Eco[pred_][v,:,zl0:zs0+1,j0:j1,i0:i1], axis=(0,1,2,3))
            predictors.append(pred_arr.copy() )
            this_name = f"avg({pred_}) t-AVG, dynamic z (ZL)"
            predictor_names.append( this_name )
            if verbose:
                print(f" {this_name}  , shape = {np.shape(pred_arr)} ,minmax={np.min(pred_arr.copy())} - {np.max(pred_arr.copy())}")

            pred_arr = np.zeros( (nv) )
            for v in np.arange( nv ):
                zs0,zl0 = zs[v,t], zs[v,t]
                pred_arr[v] = np.mean( Eco[pred_][v,:,zl0:zs0+1,j0:j1,i0:i1], axis=(0,1,2,3))
            predictors.append(pred_arr.copy() )
            this_name = f"avg({pred_}) t-AVG, dynamic z (ZS)"
            predictor_names.append( this_name )
            if verbose:
                print(f" {this_name}  , shape = {np.shape(pred_arr)} ,minmax={np.min(pred_arr.copy())} - {np.max(pred_arr.copy())}")

            pred_arr = np.zeros( (nv) )
            for v in np.arange( nv ):
                zs0,zl0 = nz_v-1, zs[v,t]
                pred_arr[v] = np.mean( Eco[pred_][v,:,zl0:zs0+1,j0:j1,i0:i1], axis=(0,1,2,3))
            predictors.append(pred_arr.copy() )
            this_name = f"avg({pred_}) t-AVG, dynamic z (0-ZS avg)"
            predictor_names.append( this_name )
            if verbose:
                print(f" {this_name}  , shape = {np.shape(pred_arr)} ,minmax={np.min(pred_arr.copy())} - {np.max(pred_arr.copy())}")


    
    print( f"Created predictor set for dynamic steer and launch " )


    if i1 > 0:
        i1p=i1-1
    else:
        i1p=nx_v+i1-1
    if j1 > 0:
        j1p=j1-1
    else:
        j1p=ny_v+j1-1
        
    filedesc = eio.make_base_filename(Eco)
    short_desc=f"Window size:( {nt_v},{ny_v},{nx_v} ), Sub-window: T({start_t},{stop_t-1}),Y({j0},{j1p}), X({i0},{i1p}) \n {key_z}"
    short_desc=short_desc + '\n' + filedesc
    
    n = 4
    predlist = '\n'.join(',  '.join(predictor_names[i:i+n]) for i in range(0, len(predictor_names), n))
    short_desc=short_desc + '\n' + '\n' + predlist

    return predictors,predictor_names,use_predictors,key_z,short_desc,zs,zl

##########################################################################
def set_A_MM_genl(Eco=None,zlev=None,use_predictors=None,
                 use_MM_winds=False, 
                 trange=None,yrange=None,xrange=None):
    """
    Only Eco and zlev are mandatory.
    Other defaults give same result as set_A above.
    """
    
    nv,nt_v,nz_v,ny_v,nx_v = np.shape( Eco.zeta_4D )

    z0=np.argmin( np.abs( zlev-0.))
    z0p5=np.argmin( np.abs( zlev-500))
    z1=np.argmin( np.abs( zlev-1000.))
    z3=np.argmin( np.abs( zlev-3000.))
    z5=np.argmin( np.abs( zlev-5000.))
    z6=np.argmin( np.abs( zlev-6000.))
    z7=np.argmin( np.abs( zlev-7000.))
    z8=np.argmin( np.abs( zlev-8000.))
    z10=np.argmin( np.abs( zlev-10000.))
    z11=np.argmin( np.abs( zlev-11000.))
    z12=np.argmin( np.abs( zlev-12000.))
    z15=np.argmin( np.abs( zlev-15000.))
    
    predictors=[]
    predictor_names=[]


    if trange is None:
        start_t, stop_t =  0 , nt_v # No Mem=> nt_v-1, nt_v 
    else:
        start_t, stop_t = trange
    if yrange is None:
        j0,j1= 0,6  
    else:
        j0,j1= yrange         
    if xrange is None:
        i0,i1= 0,nx_v  # -1 here is not eqv  - Note, Claude says use 'None'
    else:
        i0,i1= xrange 
    
    pred_scaling=1.
    targ_scaling=1.
    # use_predictors=['u_4D','v_4D','tilt_4D']

    if use_predictors is None:
        use_predictors = ['u_4D', 'v_4D', 'tilt_4D', 'zeta_4D']

    #use_predictors = ['tilt_4D', 'zeta_4D']
    
    #key_z = [z0p5, z1, z3, z5, z6, z7, z10, z11, z12, z15]
    key_z = [z0p5, z1, z3, z5, z6, z7, z10]
    for t in np.arange(start=start_t,stop=stop_t ):
        for z in key_z:
            for prd in use_predictors:
                predictors.append(np.mean(Eco[prd][:,t,z,j0:j1,i0:i1], axis=(1,2)))
                predictor_names.append(f"{prd} t={t-nt_v+1}, z={zlev[z]:.0f}")


    if use_MM_winds==True:
        # Add "MM-ey" wind calc. 
        zsteer=z3
        zlnchs=[z5,z7,z8,z10,z11]
        for zlnch in zlnchs:
            for t in np.arange(start=start_t,stop=stop_t ):
                u_steer=np.mean(Eco['u_4D'][:,t,zsteer,j0:j1,i0:i1], axis=(1,2))
                v_steer=np.mean(Eco['v_4D'][:,t,zsteer,j0:j1,i0:i1], axis=(1,2))
                u_lnch= np.mean(Eco['u_4D'][:,t,zlnch,j0:j1,i0:i1], axis=(1,2)) - u_steer
                v_lnch= np.mean(Eco['v_4D'][:,t,zlnch,j0:j1,i0:i1], axis=(1,2)) - v_steer
                usrc_mm = np.sqrt( u_lnch**2 + v_lnch**2 )
                predictors.append(usrc_mm)
                predictor_names.append(f"U_wv_src_mm t={t-nt_v+1}, zsteer:{zlev[zsteer]:.0f} zlnch:{zlev[zlnch]:.0f}")
        zsteer=z5
        zlnchs=[z7,z8,z10,z11]
        for zlnch in zlnchs:
            for t in np.arange(start=start_t,stop=stop_t ):
                u_steer=np.mean(Eco['u_4D'][:,t,zsteer,j0:j1,i0:i1], axis=(1,2))
                v_steer=np.mean(Eco['v_4D'][:,t,zsteer,j0:j1,i0:i1], axis=(1,2))
                u_lnch= np.mean(Eco['u_4D'][:,t,zlnch,j0:j1,i0:i1], axis=(1,2)) - u_steer
                v_lnch= np.mean(Eco['v_4D'][:,t,zlnch,j0:j1,i0:i1], axis=(1,2)) - v_steer
                usrc_mm = np.sqrt( u_lnch**2 + v_lnch**2 )
                predictors.append(usrc_mm)
                predictor_names.append(f"U_wv_src_mm t={t-nt_v+1}, zsteer:{zlev[zsteer]:.0f} zlnch:{zlev[zlnch]:.0f}")
        zsteer=z7
        zlnchs=[z8,z10,z11]
        for zlnch in zlnchs:
            for t in np.arange(start=start_t,stop=stop_t ):
                u_steer=np.mean(Eco['u_4D'][:,t,zsteer,j0:j1,i0:i1], axis=(1,2))
                v_steer=np.mean(Eco['v_4D'][:,t,zsteer,j0:j1,i0:i1], axis=(1,2))
                u_lnch= np.mean(Eco['u_4D'][:,t,zlnch,j0:j1,i0:i1], axis=(1,2)) - u_steer
                v_lnch= np.mean(Eco['v_4D'][:,t,zlnch,j0:j1,i0:i1], axis=(1,2)) - v_steer
                usrc_mm = np.sqrt( u_lnch**2 + v_lnch**2 )
                predictors.append(usrc_mm)
                predictor_names.append(f"U_wv_src_mm t={t-nt_v+1}, zsteer:{zlev[zsteer]:.0f} zlnch:{zlev[zlnch]:.0f}")


    
    print( f"Created predictor set=A w/ MM charcteristics " )


    if i1 > 0:
        i1p=i1-1
    else:
        i1p=nx_v+i1-1
    if j1 > 0:
        j1p=j1-1
    else:
        j1p=ny_v+j1-1
        
    filedesc = eio.make_base_filename(Eco)
    short_desc=f"Window size:( {nt_v},{ny_v},{nx_v} ), Sub-window: T({start_t},{stop_t-1}),Y({j0},{j1p}), X({i0},{i1p})"
    short_desc=short_desc + '\n' + filedesc
    
    n = 4
    predlist = '\n'.join(',  '.join(predictor_names[i:i+n]) for i in range(0, len(predictor_names), n))
    short_desc=short_desc + '\n' + '\n' + predlist

    return predictors,predictor_names,use_predictors,key_z,short_desc


def small_set_A(Eco=None,zlev=None):
    
    nv,nt_v,nz_v,ny_v,nx_v = np.shape( Eco.zeta_4D )

    z0=np.argmin( np.abs( zlev-0.))
    z0p5=np.argmin( np.abs( zlev-500))
    z1=np.argmin( np.abs( zlev-1000.))
    z3=np.argmin( np.abs( zlev-3000.))
    z5=np.argmin( np.abs( zlev-5000.))
    z6=np.argmin( np.abs( zlev-6000.))
    z7=np.argmin( np.abs( zlev-7000.))
    z10=np.argmin( np.abs( zlev-10000.))
    z11=np.argmin( np.abs( zlev-11000.))
    z12=np.argmin( np.abs( zlev-12000.))
    z15=np.argmin( np.abs( zlev-15000.))
    
    predictors=[]
    predictor_names=[]
    
    pred_scaling=1.
    targ_scaling=1.
    # use_predictors=['u_4D','v_4D','tilt_4D']
    #use_predictors = ['u_4D', 'v_4D', 'tilt_4D', 'zeta_4D']
    #use_predictors = ['u_4D', 'v_4D', 'fgf_4D', 'zeta_4D']
    
    #key_z = [z0p5, z1, z3, z5, z6, z7, z10, z11, z12, z15]
    key_z = [z10,z3]
    prd='tilt_4D'
    for t in np.arange(nt_v):
        predictors.append(np.mean(Eco[prd][:,t,z10:z3,0:6,:], axis=(1,2,3)))
        predictor_names.append(f"{prd} t={t-nt_v+1}, z_avg:{zlev[z3]:.0f}-{zlev[z10]:.0f}")
    prd='zeta_4D'
    for t in np.arange(nt_v):
        predictors.append(np.mean(Eco[prd][:,t,z10:z3,0:6,:], axis=(1,2,3)))
        predictor_names.append(f"{prd} t={t-nt_v+1}, z_avg:{zlev[z3]:.0f}-{zlev[z10]:.0f}")
    zpred=z10
    prd='u_4D'
    for t in np.arange(nt_v):
        predictors.append(np.mean(Eco[prd][:,t,zpred,0:6,:], axis=(1,2)))
        predictor_names.append(f"{prd} t={t-nt_v+1}, z:{zlev[zpred]:.0f}")
    prd='v_4D'
    for t in np.arange(nt_v):
        predictors.append(np.mean(Eco[prd][:,t,zpred,0:6,:], axis=(1,2)))
        predictor_names.append(f"{prd} t={t-nt_v+1}, z:{zlev[zpred]:.0f}")
    zpred=z3
    prd='u_4D'
    for t in np.arange(nt_v):
        predictors.append(np.mean(Eco[prd][:,t,zpred,0:6,:], axis=(1,2)))
        predictor_names.append(f"{prd} t={t-nt_v+1}, z:{zlev[zpred]:.0f}")
    prd='v_4D'
    for t in np.arange(nt_v):
        predictors.append(np.mean(Eco[prd][:,t,zpred,0:6,:], axis=(1,2)))
        predictor_names.append(f"{prd} t={t-nt_v+1}, z:{zlev[zpred]:.0f}")

    print( f"Created predictor set=smallSetA" )

    use_predictors=predictor_names
    
    return predictors,predictor_names,use_predictors,key_z
    
def small_MM_set(Eco=None,zlev=None):
    
    nv,nt_v,nz_v,ny_v,nx_v = np.shape( Eco.zeta_4D )

    z0=np.argmin( np.abs( zlev-0.))
    z0p5=np.argmin( np.abs( zlev-500))
    z1=np.argmin( np.abs( zlev-1000.))
    z3=np.argmin( np.abs( zlev-3000.))
    z5=np.argmin( np.abs( zlev-5000.))
    z6=np.argmin( np.abs( zlev-6000.))
    z7=np.argmin( np.abs( zlev-7000.))
    z8=np.argmin( np.abs( zlev-8000.))
    z10=np.argmin( np.abs( zlev-10000.))
    z11=np.argmin( np.abs( zlev-11000.))
    z12=np.argmin( np.abs( zlev-12000.))
    z15=np.argmin( np.abs( zlev-15000.))
    
    predictors=[]
    predictor_names=[]
    
    pred_scaling=1.
    targ_scaling=1.
    # use_predictors=['u_4D','v_4D','tilt_4D']
    #use_predictors = ['u_4D', 'v_4D', 'tilt_4D', 'zeta_4D']
    #use_predictors = ['u_4D', 'v_4D', 'fgf_4D', 'zeta_4D']
    
    #key_z = [z0p5, z1, z3, z5, z6, z7, z10, z11, z12, z15]
    key_z = [z10,z3]
    """
    prd='tilt_4D'
    for t in np.arange(nt_v):
        predictors.append(np.mean(Eco[prd][:,t,z10:z3,0:6,:], axis=(1,2,3)))
        predictor_names.append(f"{prd} t={t-nt_v+1}, z_avg:{zlev[z3]:.0f}-{zlev[z10]:.0f}")
    prd='tilt_4D'
    for t in np.arange(nt_v):
        predictors.append(np.mean(Eco[prd][:,t,z5:z3,0:6,:], axis=(1,2,3)))
        predictor_names.append(f"{prd} t={t-nt_v+1}, z_avg:{zlev[z3]:.0f}-{zlev[z5]:.0f}")
    prd='zeta_4D'
    for t in np.arange(nt_v):
        predictors.append(np.mean(Eco[prd][:,t,z10:z3,0:6,:], axis=(1,2,3)))
        predictor_names.append(f"{prd} t={t-nt_v+1}, z_avg:{zlev[z3]:.0f}-{zlev[z10]:.0f}")
    """
    zsteer=z3
    zlnchs=[z5,z7,z8,z10,z11]
    for zlnch in zlnchs:
        for t in np.arange(nt_v):
            u_steer=np.mean(Eco['u_4D'][:,t,zsteer,0:6,:], axis=(1,2))
            v_steer=np.mean(Eco['v_4D'][:,t,zsteer,0:6,:], axis=(1,2))
            u_lnch= np.mean(Eco['u_4D'][:,t,zlnch,0:6,:], axis=(1,2)) - u_steer
            v_lnch= np.mean(Eco['v_4D'][:,t,zlnch,0:6,:], axis=(1,2)) - v_steer
            usrc_mm = np.sqrt( u_lnch**2 + v_lnch**2 )
            predictors.append(usrc_mm)
            predictor_names.append(f"U_wv_src_mm t={t-nt_v+1}, zsteer:{zlev[zsteer]:.0f} zlnch:{zlev[zlnch]:.0f}")
    zsteer=z5
    zlnchs=[z7,z8,z10,z11]
    for zlnch in zlnchs:
        for t in np.arange(nt_v):
            u_steer=np.mean(Eco['u_4D'][:,t,zsteer,0:6,:], axis=(1,2))
            v_steer=np.mean(Eco['v_4D'][:,t,zsteer,0:6,:], axis=(1,2))
            u_lnch= np.mean(Eco['u_4D'][:,t,zlnch,0:6,:], axis=(1,2)) - u_steer
            v_lnch= np.mean(Eco['v_4D'][:,t,zlnch,0:6,:], axis=(1,2)) - v_steer
            usrc_mm = np.sqrt( u_lnch**2 + v_lnch**2 )
            predictors.append(usrc_mm)
            predictor_names.append(f"U_wv_src_mm t={t-nt_v+1}, zsteer:{zlev[zsteer]:.0f} zlnch:{zlev[zlnch]:.0f}")
    zsteer=z7
    zlnchs=[z8,z10,z11]
    for zlnch in zlnchs:
        for t in np.arange(nt_v):
            u_steer=np.mean(Eco['u_4D'][:,t,zsteer,0:6,:], axis=(1,2))
            v_steer=np.mean(Eco['v_4D'][:,t,zsteer,0:6,:], axis=(1,2))
            u_lnch= np.mean(Eco['u_4D'][:,t,zlnch,0:6,:], axis=(1,2)) - u_steer
            v_lnch= np.mean(Eco['v_4D'][:,t,zlnch,0:6,:], axis=(1,2)) - v_steer
            usrc_mm = np.sqrt( u_lnch**2 + v_lnch**2 )
            predictors.append(usrc_mm)
            predictor_names.append(f"U_wv_src_mm t={t-nt_v+1}, zsteer:{zlev[zsteer]:.0f} zlnch:{zlev[zlnch]:.0f}")

    print( f"Created predictor set=small_MM_set" )

    use_predictors=predictor_names
    
    return predictors,predictor_names,use_predictors,key_z

#==================================================================================
def GWP_like_set(Eco=None,zlev=None,
                 trange=None,yrange=None,xrange=None,
                 use_predictors=None, 
                 time_average=False):

    
    nv,nt_v,nz_v,ny_v,nx_v = np.shape( Eco.zeta_4D )

    if trange is None:
        start_t, stop_t =  0 , nt_v # No Mem=> nt_v-1, nt_v 
    else:
        start_t, stop_t = trange
    if yrange is None:
        j0,j1= 1,ny_v-1  
    else:
        j0,j1= yrange         
    if xrange is None:
        i0,i1= 0,nx_v  # -1 here is not eqv  - Note, Claude says use 'None'
    else:
        i0,i1= xrange 

    z0=np.argmin( np.abs( zlev-0.))
    z0p5=np.argmin( np.abs( zlev-500))
    z1=np.argmin( np.abs( zlev-1000.))
    z3=np.argmin( np.abs( zlev-3000.))
    z5=np.argmin( np.abs( zlev-5000.))
    z6=np.argmin( np.abs( zlev-6000.))
    z7=np.argmin( np.abs( zlev-7000.))
    z8=np.argmin( np.abs( zlev-8000.))
    z10=np.argmin( np.abs( zlev-10000.))
    z11=np.argmin( np.abs( zlev-11000.))
    z12=np.argmin( np.abs( zlev-12000.))
    z15=np.argmin( np.abs( zlev-15000.))
    
    predictors=[]
    predictor_names=[]
    
    z_limits = [ [z3,z10],[z0,z3] ]

    if use_predictors == None:
        use_predictors = ['tilt_4D','stab_4D']

    climo={'stab_4D':0.,'zeta_4D':0.,'tilt_4D':2.e-7,}
    if time_average==True:
        for prd in use_predictors:
            for zran in z_limits:
                zbot,ztop = zran
                this_pred = np.mean(Eco[prd][:, start_t:stop_t, ztop:zbot, j0:j1, i0:i1], axis=(1,2,3,4))  #-climo[prd] # this doens't seem successful
                predictors.append(this_pred)
                predictor_names.append(f"{prd} t_avg={start_t-nt_v+1},{stop_t-nt_v} ; z_avg={zlev[zbot]:.0f}-{zlev[ztop]:.0f}")

                
    
        
        
        # Add "MM-ey" wind calc. 
        zsteer=z3
        zlnchs=[z10,]
        for zlnch in zlnchs:
            u_steer=np.mean(Eco['u_4D'][:,start_t:stop_t,zsteer,j0:j1,i0:i1], axis=(1,2,3))
            v_steer=np.mean(Eco['v_4D'][:,start_t:stop_t,zsteer,j0:j1,i0:i1], axis=(1,2,3))
            u_lnch= np.mean(Eco['u_4D'][:,start_t:stop_t,zlnch,j0:j1,i0:i1], axis=(1,2,3)) - u_steer
            v_lnch= np.mean(Eco['v_4D'][:,start_t:stop_t,zlnch,j0:j1,i0:i1], axis=(1,2,3)) - v_steer
            usrc_mm = np.sqrt( u_lnch**2 + v_lnch**2 )
            predictors.append(usrc_mm)
            predictor_names.append(f"U_wv_src_mm t_avg={start_t-nt_v+1},{stop_t-nt_v}; zsteer={zlev[zsteer]:.0f}, zlnch={zlev[zlnch]:.0f}")
       

    else:
        for prd in use_predictors:
            for t in np.arange(start_t, stop_t):  # correct
                for zran in z_limits:
                    zbot,ztop = zran
                    predictors.append(np.mean(Eco[prd][:,t,ztop:zbot,j0:j1,i0:i1], axis=(1,2,3)))
                    predictor_names.append(f"{prd} t={t-nt_v+1}, z_avg:{zlev[zbot]:.0f}-{zlev[ztop]:.0f}")
    
        
        
        # Add "MM-ey" wind calc. 
        zsteer=z3
        zlnchs=[z10,]
        for zlnch in zlnchs:
            for t in np.arange(start=start_t,stop=stop_t ):
                u_steer=np.mean(Eco['u_4D'][:,t,zsteer,j0:j1,i0:i1], axis=(1,2))
                v_steer=np.mean(Eco['v_4D'][:,t,zsteer,j0:j1,i0:i1], axis=(1,2))
                u_lnch= np.mean(Eco['u_4D'][:,t,zlnch,j0:j1,i0:i1], axis=(1,2)) - u_steer
                v_lnch= np.mean(Eco['v_4D'][:,t,zlnch,j0:j1,i0:i1], axis=(1,2)) - v_steer
                usrc_mm = np.sqrt( u_lnch**2 + v_lnch**2 )
                predictors.append(usrc_mm)
                predictor_names.append(f"U_wv_src_mm t={t-nt_v+1}, zsteer:{zlev[zsteer]:.0f} zlnch:{zlev[zlnch]:.0f}")
   
    
    
    if i1 > 0:
        i1p=i1-1
    else:
        i1p=nx_v+i1-1
    if j1 > 0:
        j1p=j1-1
    else:
        j1p=ny_v+j1-1
        
    filedesc = eio.make_base_filename(Eco)
    short_desc=f"Window size:( {nt_v},{ny_v},{nx_v} ), Sub-window: T({start_t},{stop_t-1}),Y({j0},{j1p}), X({i0},{i1p}) ... 'GWP-like set' "
    short_desc=short_desc + '\n' + filedesc
    
    n = 4
    predlist = '\n'.join(',  '.join(predictor_names[i:i+n]) for i in range(0, len(predictor_names), n))
    short_desc=short_desc + '\n' + '\n' + predlist
    
    return predictors,predictor_names,use_predictors,z_limits,short_desc
#==================================================================================
def MovMtn_like_set(Eco=None,zlev=None,z_steer=None,z_launch=None,
                 trange=None,yrange=None,xrange=None,
                 use_predictors=None, 
                 time_average=False):

    
    nv,nt_v,nz_v,ny_v,nx_v = np.shape( Eco.zeta_4D )

    if trange is None:
        start_t, stop_t =  0 , nt_v # No Mem=> nt_v-1, nt_v 
    else:
        start_t, stop_t = trange
    if yrange is None:
        j0,j1= 1,ny_v-1  
    else:
        j0,j1= yrange         
    if xrange is None:
        i0,i1= 0,nx_v  # -1 here is not eqv  - Note, Claude says use 'None'
    else:
        i0,i1= xrange 

    zst=np.argmin( np.abs( zlev-z_steer ))
    zln=np.argmin( np.abs( zlev-z_launch ))
    
    predictors=[]
    predictor_names=[]
    
    if use_predictors == None:
        use_predictors = ['zeta_4D',]

    if time_average==True:
        for prd in use_predictors:
            this_pred = np.mean(Eco[prd][:, start_t:stop_t, zln:zst, j0:j1, i0:i1], axis=(1,2,3,4))  #-climo[prd] # this doens't seem successful
            predictors.append(this_pred)
            predictor_names.append(f"{prd} t_avg={start_t-nt_v+1},{stop_t-nt_v} ; z_avg={zlev[zst]:.0f}-{zlev[zln]:.0f}")
        
        # Add "MM-ey" wind calc. 
        u_steer=np.mean(Eco['u_4D'][:,start_t:stop_t,zst,j0:j1,i0:i1], axis=(1,2,3))
        v_steer=np.mean(Eco['v_4D'][:,start_t:stop_t,zst,j0:j1,i0:i1], axis=(1,2,3))
        u_lnch= np.mean(Eco['u_4D'][:,start_t:stop_t,zln,j0:j1,i0:i1], axis=(1,2,3)) - u_steer
        v_lnch= np.mean(Eco['v_4D'][:,start_t:stop_t,zln,j0:j1,i0:i1], axis=(1,2,3)) - v_steer
        usrc_mm = np.sqrt( u_lnch**2 + v_lnch**2 )
        predictors.append(usrc_mm)
        predictor_names.append(f"U_wv_src_mm t_avg={start_t-nt_v+1},{stop_t-nt_v}; zsteer={zlev[zst]:.0f}, zlnch={zlev[zln]:.0f}")
       

    else:
        for prd in use_predictors:
            for t in np.arange(start_t, stop_t):  # correct
                predictors.append(np.mean(Eco[prd][:,t,zln:zst,j0:j1,i0:i1], axis=(1,2,3)))
                predictor_names.append(f"{prd} t={t-nt_v+1}, z_avg:{zlev[zst]:.0f}-{zlev[zln]:.0f}")
    
        # Add "MM-ey" wind calc. 
        for t in np.arange(start=start_t,stop=stop_t ):
            u_steer=np.mean(Eco['u_4D'][:,t,zst,j0:j1,i0:i1], axis=(1,2))
            v_steer=np.mean(Eco['v_4D'][:,t,zst,j0:j1,i0:i1], axis=(1,2))
            u_lnch= np.mean(Eco['u_4D'][:,t,zln,j0:j1,i0:i1], axis=(1,2)) - u_steer
            v_lnch= np.mean(Eco['v_4D'][:,t,zln,j0:j1,i0:i1], axis=(1,2)) - v_steer
            usrc_mm = np.sqrt( u_lnch**2 + v_lnch**2 )
            predictors.append(usrc_mm)
            predictor_names.append(f"U_wv_src_mm t={t-nt_v+1}, zsteer:{zlev[zst]:.0f} zlnch:{zlev[zln]:.0f}")
   
    
    
    if i1 > 0:
        i1p=i1-1
    else:
        i1p=nx_v+i1-1
    if j1 > 0:
        j1p=j1-1
    else:
        j1p=ny_v+j1-1
        
    filedesc = eio.make_base_filename(Eco)
    short_desc=f"Window size:( {nt_v},{ny_v},{nx_v} ), Sub-window: T({start_t},{stop_t-1}),Y({j0},{j1p}), X({i0},{i1p}) ... 'MovMtn-like set' "
    short_desc=short_desc + '\n' + filedesc
    
    n = 4
    predlist = '\n'.join(',  '.join(predictor_names[i:i+n]) for i in range(0, len(predictor_names), n))
    short_desc=short_desc + '\n' + '\n' + predlist
    z_limits = [ [zst,zln] ]

    return predictors,predictor_names,use_predictors,z_limits,short_desc
