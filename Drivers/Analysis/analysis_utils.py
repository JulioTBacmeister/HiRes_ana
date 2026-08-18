import sys
import numpy as np

from scipy.ndimage import label
from scipy import ndimage as ndi
from scipy import stats

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt
import event_utils as euti

from Utils import MyConstants as Co
grav=Co.grav()

# This allows for both dict.key and dict['key'] syntax
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


def find_gw_events(epwp, thresh, connectivity=8):
    """
    Identify gravity-wave events in a 2D momentum-flux field.

    Parameters
    ----------
    epwp : 2D numpy array (ny,nx)
        Momentum flux field.
    thresh : float
        Threshold for event detection.
    connectivity : int
        4 or 8 for pixel connectivity.

    Returns
    -------
    events : list of dict
        Each event contains:
            iy, ix : location of max flux
            epwp_max : maximum flux
            size : number of grid cells
            epwp_sum : integrated flux in event
    """

    #print( f"Threhold size {np.size(thresh)}" )
    if np.size(thresh)==1:
        mask = epwp > thresh
    elif np.size(thresh)==2:
        mask = (epwp>thresh[0]) & (epwp<=thresh[1])
    else:
        raise ValueError("threshold missing or wrong size")
    
    if connectivity == 8:
        structure = np.ones((3,3), dtype=int)
    else:
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]])

    labels, n_events = label(mask, structure=structure)

    events = []

    for i in range(1, n_events+1):

        region = (labels == i)

        iy, ix = np.where(region)
        vals = epwp[iy, ix]

        k = np.argmax(vals)

        event = {
            "iy": iy[k],
            "ix": ix[k],
            "epwp_max": vals[k],
            "size": len(vals),
            "epwp_sum": vals.sum(),
        }

        events.append(event)

    return events

def find_gw_events_watershed_2(epwp, thresh, use_watershed=True,  use_peak_local_max=True, second_thresh=None, 
                              lat_range=None, lon_range=None, lat=None, lon=None, peak_footprint=(3,3) ):

    if np.size(thresh)==1:
        mask = epwp > thresh
    elif np.size(thresh)==2:
        mask = (epwp > thresh[0]) & (epwp <= thresh[1])
    else:
        raise ValueError("threshold missing or wrong size")

    # footprint=np.ones((3,3)),

    #print( f"using peak_footprint {peak_footprint}")
    if use_peak_local_max:
        peaks = peak_local_max(
            epwp,
            footprint=np.ones( peak_footprint ),
            labels=mask
        )
    else:
        iy, ix = np.where(mask)
        peaks = np.column_stack((iy, ix))

    events = []

    if use_watershed:
        markers = np.zeros_like(epwp, dtype=int)

        for i, (y, x) in enumerate(peaks, start=1):
            markers[y, x] = i

        labels = watershed(-epwp, markers, mask=mask)

        for i in range(1, labels.max() + 1):
            region = labels == i

            iy, ix = np.where(region)
            vals = epwp[iy, ix]

            k = np.argmax(vals)

            events.append({
                "iy": iy[k],
                "ix": ix[k],
                "epwp_max": vals[k],
                "size": len(vals),
                "epwp_sum": vals.sum()
            })

    else:
        # treat each detected peak as an event directly
        for (y, x) in peaks:
            events.append({
                "iy": y,
                "ix": x,
                "epwp_max": epwp[y, x],
                "size": 1,
                "epwp_sum": epwp[y, x]
            })


    #####################################
    # Secondary culling ....
    #####################################
    if second_thresh is not None:
        thr0,thr1=second_thresh
        events_x=[]
        for e in events:
            if e['epwp_max'] >thr0 and e['epwp_max']<=thr1:
                events_x.append( e )
        events = events_x
        
    if (lat_range is not None) and (lat is not None):
        latS,latN=lat_range[0],lat_range[1]
        events_x=[]
        for e in events:
            lat0 = lat[int(e["iy"])]
            if (lat0>=latS) and (lat0<=latN):
                events_x.append( e )
        events = events_x

    if (lon_range is not None) and (lon is not None):
        lonW,lonE=lon_range[0],lon_range[1]
        events_x=[]
        for e in events:
            lon0 = lon[int(e["ix"])]
            if (lon0>=lonW) and (lon0<=lonE):
                events_x.append( e )
        events = events_x

    
    return events


def find_gw_events_watershed(epwp, thresh):

    #print( f"Threhold size {np.size(thresh)}" )
    if np.size(thresh)==1:
        mask = epwp > thresh
    elif np.size(thresh)==2:
        mask = (epwp>thresh[0]) & (epwp<=thresh[1])
    else:
        raise ValueError("threshold missing or wrong size")
    #print( f" size of True - epwp>{thresh} = {np.sum(mask)}  " )
    #print( f" size of True mask = {np.sum(mask)} " )

    
    # distance field helps watershed separate peaks
    distance = ndi.distance_transform_edt(mask)

    # find local maxima of momentum flux
    peaks = peak_local_max(
        epwp,
        #footprint=np.ones((5,5)),
        footprint=np.ones( (3,3) ),
        labels=mask
    )

    markers = np.zeros_like(epwp, dtype=int)

    for i, (y, x) in enumerate(peaks, start=1):
        markers[y, x] = i

    labels = watershed(-epwp, markers, mask=mask)

    events = []

    count = 0
    for i in range(1, labels.max()+1):

        region = labels == i

        iy, ix = np.where(region)
        vals = epwp[iy, ix]

        k = np.argmax(vals)

        events.append({
            "iy": iy[k],
            "ix": ix[k],
            "epwp_max": vals[k],
            "size": len(vals),
            "epwp_sum": vals.sum()
        })
        count=count+1

    #print( f" Culled down to {count} 'events'" )
        
    return events

def cube4D( event_list, aa , lat, lon, window=[5,5,5] , TZHkey='tzyx', lat_range=[-90,90], lon_range=[0,360] ):
    #################################################################################################################
    #  Returns a 4D (note vertical dim is constant) picture of input aa around events in event_list. 
    #  Size of picture is determined by window argument, where window=[wt,wy,wx] ; itime,ilat,ilon resp.

    nt_e = len( event_list )

    latS,latN=lat_range[0],lat_range[1]
    lonW,lonE=lon_range[0],lon_range[1]

    
    if TZHkey == 'tzyx':
        nt_aa,nz,ny,nx = np.shape( aa )
    elif TZHkey == 'tyx':
        nt_aa,ny,nx = np.shape( aa )
    else:
        print( "Not set up for this array shape " )
        return -999

    if (nt_e != nt_aa ):
        print( "Inconsistent time in events and array" )
        return -999
        
    count=0
    #for evs in event_list:
    #    for ev in evs:
    for t in np.arange(nt_e):
        evs = event_list[t]
        nev = len( evs )
        for e in np.arange( nev ):
            ev = evs[e]
            lat0=lat[ ev['iy'] ]
            lon0=lon[ ev['ix'] ]
            if (lat0>=latS) and (lat0<=latN) and (lon0>=lonW) and (lon0<=lonE):
                count=count+1
    
    #print( count )





    wt,wy,wx = window
    st = 2 # if subsample_time else 1  # step of 2 = every other index
    wt_c = (wt // st) + 1  # number of samples taken

    print( f' in auti wt_c {wt_c}' )
    
    if TZHkey == 'tzyx':
        aa_pad = np.pad(aa, ((wt, wt), (0, 0), (wy, wy), (0, 0)), mode='edge')
        aa_pad = np.pad(aa_pad, ((0, 0), (0, 0), (0, 0), (wx, wx)), mode='wrap')
        aa_comp = np.zeros( ( count , wt_c+1, nz, 2*wy+1, 2*wx+1 )  )
    elif TZHkey == 'tyx':
        aa_pad = np.pad(aa, ((wt, wt), (wy, wy), (0, 0)), mode='edge')
        aa_pad = np.pad(aa_pad, ((0, 0), (0, 0), (wx, wx)), mode='wrap')
        aa_comp = np.zeros( ( count , wt_c+1, 2*wy+1, 2*wx+1 )  )
    c=0
    #for evs in event_list:
    #    for ev in evs:
    for t in np.arange(nt_e):
        evs = event_list[t]
        nev = len( evs )
        for e in np.arange( nev ):
            ev = evs[e]
            lat0=lat[ ev['iy'] ]
            lon0=lon[ ev['ix'] ]
            if (lat0>=latS) and (lat0<=latN) and (lon0>=lonW) and (lon0<=lonE):
                y=ev['iy'] + wy
                x=ev['ix'] + wx
                t_ = t + wt
                if TZHkey == 'tzyx':
                    #aa_comp[c, 0:wt+1, :,0:2*wy+1, 0:2*wx+1 ] = aa_pad[t_-wt:t_+1, :, y-wy:y+wy+1, x-wx:x+wx+1 ] 
                    aa_comp[c, 0:wt_c, :, 0:2*wy+1, 0:2*wx+1] =  aa_pad[t_-wt:t_+1:st, :, y-wy:y+wy+1, x-wx:x+wx+1]                    
                elif TZHkey == 'tyx':
                    aa_comp[c, 0:wt+1, 0:2*wy+1, 0:2*wx+1 ] = aa_pad[t_-wt:t_+1:st ,y-wy:y+wy+1,x-wx:x+wx+1]                 
                c=c+1
    
    #print( c )
    
    
    #for t in np.arange( nt ):
    #    for ie in np.arange( nevs ):
    return aa_comp

def cube4D_ds( event_ds, aa , lat, lon, window=[5,5,5] , TZHkey='tzyx', lat_range=[-90,90], lon_range=[0,360], mask2D=None, subsample_time=False ):
    #################################################################################################################
    #  Returns a 4D (note vertical dim is constant) picture of input aa around events in event_list. 
    #  Size of picture is determined by window argument, where window=[wt,wy,wx] ; itime,ilat,ilon resp.

    latS,latN=lat_range[0],lat_range[1]
    lonW,lonE=lon_range[0],lon_range[1]

    
    if TZHkey == 'tzyx':
        nt_aa,nz,ny,nx = np.shape( aa )
    elif TZHkey == 'tyx':
        nt_aa,ny,nx = np.shape( aa )
    else:
        print( "Not set up for this array shape " )
        return -999
        
    count=0
    lon_event=event_ds.lon_event.values
    lat_event=event_ds.lat_event.values
    nv = len( lat_event )
    for v in np.arange(nv):
        lat0=lat_event[ v ]
        lon0=lon_event[ v ]
        if (lat0>=latS) and (lat0<=latN) and (lon0>=lonW) and (lon0<=lonE):
            count=count+1
    
    #print( count )

    #return 999999.

    wt,wy,wx = window
    st = 2 if subsample_time else 1  # step of 2 = every other index
    wt_c = (wt // st) #+ 1  # number of samples taken
    if ( st*wt_c != wt ):
        sys.exit( f"Shouldn't subsample unless wt is even: wt={wt}" )
    #print( f' in auti.cube4D_ds: wt {wt}, wt_c {wt_c}' )
    
    wt,wy,wx = window
    if TZHkey == 'tzyx':
        aa_pad = np.pad(aa, ((wt, wt), (0, 0), (wy, wy), (0, 0)), mode='edge')
        aa_pad = np.pad(aa_pad, ((0, 0), (0, 0), (0, 0), (wx, wx)), mode='wrap')
        aa_comp = np.zeros( ( count , wt_c+1, nz, 2*wy+1, 2*wx+1 )  )
    elif TZHkey == 'tyx':
        aa_pad = np.pad(aa, ((wt, wt), (wy, wy), (0, 0)), mode='edge')
        aa_pad = np.pad(aa_pad, ((0, 0), (0, 0), (wx, wx)), mode='wrap')
        aa_comp = np.zeros( ( count , wt_c+1, 2*wy+1, 2*wx+1 )  )

    time_comp = np.zeros( ( count )  )
    lat_comp = np.zeros( ( count )  )
    lon_comp = np.zeros( ( count )  )

    c=0
    ix_event=event_ds.ix.values
    iy_event=event_ds.iy.values
    it_event=event_ds.itime.values
    for v in np.arange( nv ):
        lat0=lat_event[ v ]
        lon0=lon_event[ v ]
        if (lat0>=latS) and (lat0<=latN) and (lon0>=lonW) and (lon0<=lonE):
            t=it_event[v]
            y=iy_event[v] + wy
            x=ix_event[v] + wx
            t_ = t + wt
            if TZHkey == 'tzyx':
                #aa_comp[c, 0:wt+1, :,0:2*wy+1, 0:2*wx+1 ] = aa_pad[t_-wt:t_+1, :, y-wy:y+wy+1, x-wx:x+wx+1 ] 
                aa_comp[c, 0:wt_c+1, :, 0:2*wy+1, 0:2*wx+1] =  aa_pad[t_-wt:t_+1:st, :, y-wy:y+wy+1, x-wx:x+wx+1]                    
            elif TZHkey == 'tyx':
                #aa_comp[c, 0:wt+1, 0:2*wy+1, 0:2*wx+1 ] = aa_pad[t_-wt:t_+1 ,y-wy:y+wy+1,x-wx:x+wx+1]                 
                aa_comp[c, 0:wt_c+1, 0:2*wy+1, 0:2*wx+1] =  aa_pad[t_-wt:t_+1:st, y-wy:y+wy+1, x-wx:x+wx+1]                    
            time_comp[c] , lat_comp[c] , lon_comp[c] = t, lat0, lon0
            c=c+1
    
    #print( c )

    
    reltime_raw = np.zeros( wt+1 )
    for t in np.arange( wt+1):
        reltime_raw[t] = -wt+t
    reltime_x = np.zeros( wt_c+1 )
    reltime_x = reltime_raw[ 0:wt+1:st ]

    print( f"relative time array: {reltime_x} " )
    
    #for t in np.arange( nt ):
    #    for ie in np.arange( nevs ):
    return aa_comp,time_comp,lat_comp,lon_comp
    
def cube4D_relative_time( window=[5,5,5] , subsample_time=False ):
    #################################################################################################################
    #  Returns a 4D (note vertical dim is constant) picture of input aa around events in event_list. 
    #  Size of picture is determined by window argument, where window=[wt,wy,wx] ; itime,ilat,ilon resp.


    wt,wy,wx = window
    st = 2 if subsample_time else 1  # step of 2 = every other index
    wt_c = (wt // st) #+ 1  # number of samples taken
    if ( st*wt_c != wt ):
        sys.exit( f"Shouldn't subsample unless wt is even: wt={wt}" )
    
    reltime_raw = np.zeros( wt+1 )
    for t in np.arange( wt+1):
        reltime_raw[t] = -wt+t
    reltime_x = np.zeros( wt_c+1 )
    reltime_x = reltime_raw[ 0:wt+1:st ]

    print( f"New function for relative time array: {reltime_x} " )
    
    #for t in np.arange( nt ):
    #    for ie in np.arange( nevs ):
    return reltime_x

def collapseSpace( aa_comp  , TZHkey='etzyx'  ):

    if TZHkey == 'etzyx':
        nc,nt,nz,ny,nx = np.shape( aa_comp )
        aa_comp_mean = aa_comp.mean(axis=(3,4))
        aa_comp_min  = aa_comp.min(axis=(3,4))
        aa_comp_max  = aa_comp.max(axis=(3,4))    
    elif TZHkey == 'etyx':
        nc,nt,ny,nx = np.shape( aa_comp )
        aa_comp_mean = aa_comp.mean(axis=(2,3))
        aa_comp_min  = aa_comp.min(axis=(2,3))
        aa_comp_max  = aa_comp.max(axis=(2,3))    
    else:
        print( f"no valid shape" )

    return aa_comp_mean,aa_comp_min,aa_comp_max

def collapseSpaceTime( aa_comp  , TZHkey='etzyx'  ):
    
    aa_MMM = collapseSpace( aa_comp  , TZHkey=TZHkey )
    aa_super_mean = aa_MMM[0].mean( axis=1 )
    aa_super_min = aa_MMM[1].min( axis=1 )
    aa_super_max = aa_MMM[2].max( axis=1 )

    return aa_MMM, aa_super_mean,aa_super_min, aa_super_max

def track_events(event_lists, max_dist=5):
    """
    event_lists: list over time of lists of events
                 events contain 'iy','ix'
    max_dist: maximum allowed movement (grid cells)

    returns: list of tracks
    """

    tracks = []
    active_tracks = []

    for t, events in enumerate(event_lists):

        used = set()

        # try to extend existing tracks
        for tr in active_tracks:

            y0, x0 = tr["iy"][-1], tr["ix"][-1]

            best = None
            best_dist = max_dist

            for i, ev in enumerate(events):
                if i in used:
                    continue

                dy = ev["iy"] - y0
                dx = ev["ix"] - x0
                d = np.sqrt(dy*dy + dx*dx)

                if d < best_dist:
                    best = i
                    best_dist = d

            if best is not None:
                ev = events[best]

                tr["time"].append(t)
                tr["iy"].append(ev["iy"])
                tr["ix"].append(ev["ix"])

                used.add(best)
            else:
                tracks.append(tr)

        # start new tracks
        new_active = []

        for i, ev in enumerate(events):

            if i not in used:
                new_active.append({
                    "time":[t],
                    "iy":[ev["iy"]],
                    "ix":[ev["ix"]],
                })

        active_tracks = new_active

    tracks.extend(active_tracks)

    return tracks

def ddz_var(f, z):
    """
    Vertical derivative df/dz for f(nt,nz,ny,nx) with z(nt,nz,ny,nx).

    Uses centered differences in the interior and one-sided differences
    at the top/bottom.
    """
    dfdz = np.empty_like(f)

    # bottom
    dfdz[:, 0, :, :] = (
        f[:, 1, :, :] - f[:, 0, :, :]
    ) / (
        z[:, 1, :, :] - z[:, 0, :, :]
    )

    # top
    dfdz[:, -1, :, :] = (
        f[:, -1, :, :] - f[:, -2, :, :]
    ) / (
        z[:, -1, :, :] - z[:, -2, :, :]
    )

    # interior: centered
    dfdz[:, 1:-1, :, :] = (
        f[:, 2:, :, :] - f[:, :-2, :, :]
    ) / (
        z[:, 2:, :, :] - z[:, :-2, :, :]
    )

    return dfdz


def tiltmag(u, v, zeta, zm):
    """
    Compute

        tilt = || zeta * du/dz, zeta * dv/dz ||

    for arrays shaped (nt,nz,ny,nx).
    """
    du_dz = ddz_var(u, zm)
    dv_dz = ddz_var(v, zm)

    tilt = np.sqrt( (zeta * du_dz)**2 + (zeta * dv_dz)**2 )
    del du_dz,dv_dz
    return tilt

def stability(th, zm):
    """
    Compute

        tilt = || zeta * du/dz, zeta * dv/dz ||

    for arrays shaped (nt,nz,ny,nx).
    """
    dth_dz = ddz_var(th, zm)

    stab = grav * dth_dz / th 
    del dth_dz
    return stab

def one_dim_pdf(x , nbin=50, log=False, density=True ):

    x = x[np.isfinite(x)]

    if log==True:
        x = x[x > 0.0]
        bins = np.logspace(np.log10(x.min()), np.log10(x.max()), nbin + 1)
        # Bin centers
        xcens = np.sqrt(bins[:-1] * bins[1:])
    else:
        bins = np.linspace(x.min(), x.max(), nbin + 1)
        # Bin centers
        xcens = 0.5*( bins[:-1] + bins[1:] )
        
    
    # Which bin each value falls into
    ibin = np.digitize(x, bins) - 1
    
    # Sum of x within each bin
    count1d = np.zeros(nbin)
    
    for i in range(nbin):
        m = ibin == i
        count1d[i] = np.sum(x[m])
        
    if density==False:
        return count1d,xcens
    
    dx = np.diff(bins)          # shape (nxbin,)
    
    pdf1d = count1d / len(x) / dx

    return pdf1d,xcens

def two_dim_pdf(x,y,nxbin=50,nybin=50,logx=False,logy=False, density=True ):
    
    # x, y are 1D arrays of same length
    # for example: x = np.abs(upwp), y = np.abs(zeta)
    
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    
    if logx==True:
        xbins = np.logspace(np.log10(x.min()), np.log10(x.max()), nxbin + 1)
        # bin centers: geometric mean is best for log bins
        xcens = np.sqrt(xbins[:-1] * xbins[1:])
    else:
        xbins = np.linspace(x.min(), x.max(), nxbin + 1)
        xcens = 0.5*(xbins[:-1] + xbins[1:])
        
    if logy==True:
        ybins = np.logspace(np.log10(y.min()), np.log10(y.max()), nybin + 1)
        # bin centers: geometric mean is best for log bins
        ycens = np.sqrt(ybins[:-1] * ybins[1:])
    else:
        ybins = np.linspace(y.min(), y.max(), nxbin + 1)
        ycens = 0.5*(ybins[:-1] + ybins[1:])
    
    ibin = np.digitize(x, xbins) - 1
    jbin = np.digitize(y, ybins) - 1
    
    # protect against points on the upper edge
    ibin = np.clip(ibin, 0, nxbin - 1)
    jbin = np.clip(jbin, 0, nybin - 1)
    
    count2d = np.zeros((nxbin, nybin))
    
    for i in range(nxbin):
        for j in range(nybin):
            m = (ibin == i) & (jbin == j)
            count2d[i, j] = np.sum(m)

    if density==False:
        return count2d,xcens,ycens
    
    dx = np.diff(xbins)          # shape (nxbin,)
    dy = np.diff(ybins)          # shape (nybin,)
    
    area = dx[:, None] * dy[None, :]   # shape (nxbin, nybin)
    
    pdf2d = count2d / len(x) / area

    return pdf2d,xcens,ycens

def loglogpdf(epwp):


    # upwp: flat 1D ndarray
    x = epwp  #np.abs(upwp)                      # magnitudes
    x = x[np.isfinite(x)]                 # remove NaN/inf
    x = x[x > 0.0]                        # log scale cannot use zero
    
    # log-spaced bins
    nbins = 50
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), nbins + 1)
    
    # PDF estimate from histogram
    pdf, edges = np.histogram(x, bins=bins, density=True)
    
    # bin centers: geometric mean is best for log bins
    centers = np.sqrt(edges[:-1] * edges[1:])
    
    # plot
    fig, ax = plt.subplots()
    ax.plot(centers, pdf, marker='o', linestyle='-')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel('|upwp|')
    ax.set_ylabel('PDF')
    ax.set_title('Log-log PDF of |upwp|')

def contrib_to_total(epwp):

    x = epwp # np.abs(upwp)
    x = x[np.isfinite(x)]
    x = x[x > 0.0]
    
    nbins = 50
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), nbins + 1)
    
    # Which bin each value falls into
    ibin = np.digitize(x, bins) - 1
    
    # Sum of |upwp| within each bin
    bin_sum = np.zeros(nbins)
    
    for i in range(nbins):
        m = ibin == i
        bin_sum[i] = np.sum(x[m])
    
    # Fractional contribution of each bin to total |upwp|
    frac = bin_sum / np.sum(bin_sum)
    
    # Bin centers
    centers = np.sqrt(bins[:-1] * bins[1:])
    
    # Plot
    fig, ax = plt.subplots()
    
    m = frac > 0
    ax.plot(centers[m], frac[m], marker='o')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel('|upwp|')
    ax.set_ylabel('Fraction of total |upwp|')
    ax.set_title('Contribution of each |upwp| bin to total magnitude')

    

def cumul_big_to_small(epwp, plot_it=False):

    x = epwp # np.abs(upwp)
    x = x[np.isfinite(x)]
    x = x[x > 0.0]
    
    # sort ascending
    xs = np.sort(x)
    
    # cumulative sum from the top tail downward
    tail_sum = np.cumsum(xs[::-1])[::-1]
    tail_frac = tail_sum / tail_sum[0]

    if plot_it==True:
        fig, ax = plt.subplots()
        ax.plot(xs, tail_frac)
        
        ax.set_xscale('log')
        ax.set_xlabel('X')
        ax.set_ylabel(r'Fraction of total |upwp| from values >= X')
        ax.set_title(r'Tail contribution of $|upwp|$')

    return tail_frac,xs

def event_composite_correlation(A_4D, composite=None, CorrKey='tzyx'):
    """
    Compute spatial correlation of each event with the composite mean.
    
    Parameters
    ----------
    A_4D : np.ndarray, shape (n_events, n_t, n_z, n_y, n_x)
        The 4D column data for each event.
    composite : np.ndarray or None
        Precomputed composite (mean over events). If None, computed from A_4D.
        Shape should be (n_t, n_z, n_y, n_x).
    CorrKey : str
        Which dimensions to correlate over:
        'tzyx' -> scalar r per event,        output shape (n_events,)
        'tyx'  -> z profile per event,        output shape (n_events, n_z)
        'yx'   -> (t,z) array per event,      output shape (n_events, n_t, n_z)
    
    Returns
    -------
    r : np.ndarray
        Pearson r of each event against the composite.
    p : np.ndarray
        Two-tailed p-values, same shape as r.
    summary : dict
        Mean r, std r, median r, outlier threshold and indices.
        For non-scalar r, these statistics are computed over events
        at each (t,z) point.
    """
    # --- dimension mapping -------------------------------------------------
    # A_4D axes:  0=event, 1=t, 2=z, 3=y, 4=x
    key_to_corr_axes = {
        'tzyx': (1, 2, 3, 4),   # correlate all -> r shape (n_events,)
        'tyx':  (1, 3, 4),      # correlate t,y,x -> r shape (n_events, n_z)
        'yx':   (3, 4),         # correlate y,x   -> r shape (n_events, n_t, n_z)
    }
    if CorrKey not in key_to_corr_axes:
        raise ValueError(f"CorrKey must be one of {list(key_to_corr_axes.keys())}")

    corr_axes = key_to_corr_axes[CorrKey]
    loop_axes = [ax for ax in (1, 2, 3, 4) if ax not in corr_axes]  # axes we loop over

    n_events          = A_4D.shape[0]
    n_t, n_z, n_y, n_x = A_4D.shape[1:]

    if composite is None:
        composite = A_4D.mean(axis=0)  # shape (n_t, n_z, n_y, n_x)

    # --- build output shape ------------------------------------------------
    # loop_axes refer to A_4D axes 1-4; subtract 1 to index into (n_t,n_z,n_y,n_x)
    loop_shape = tuple(A_4D.shape[ax] for ax in loop_axes)  # e.g. () or (n_z,) or (n_t,n_z)
    out_shape  = (n_events,) + loop_shape                   # e.g. (n_events,) or (n_events,n_z)

    r = np.zeros(out_shape)
    p = np.zeros(out_shape)

    # --- helper: given a loop index tuple, build the slice for A_4D[e] ----
    # A_4D[e] has axes (t, z, y, x) = axes 0,1,2,3 of the sliced array
    # corr_axes in A_4D terms are 1,2,3,4 -> subtract 1 for the sliced array
    corr_axes_local = tuple(ax - 1 for ax in corr_axes)   # axes within A_4D[e]
    loop_axes_local = tuple(ax - 1 for ax in loop_axes)

    def _get_slice(arr, loop_idx):
        """Extract the sub-array for one loop-index combination, then flatten corr dims."""
        # Build a full index: slice(None) for corr axes, integer for loop axes
        idx = [slice(None)] * 4
        for ax, i in zip(loop_axes_local, loop_idx):
            idx[ax] = i
        return arr[tuple(idx)].flatten()

    # --- iterate over loop index combinations ------------------------------
    import itertools
    loop_sizes    = [A_4D.shape[ax] for ax in loop_axes]
    loop_indices  = list(itertools.product(*[range(s) for s in loop_sizes]))

    if not loop_indices:          # CorrKey='tzyx': single scalar per event
        loop_indices = [()]

    for loop_idx in loop_indices:
        comp_vec = _get_slice(composite, loop_idx)
        for e in range(n_events):
            event_vec = _get_slice(A_4D[e], loop_idx)
            r_val, p_val = stats.pearsonr(event_vec, comp_vec)
            if loop_idx:
                r[(e,) + loop_idx] = r_val
                p[(e,) + loop_idx] = p_val
            else:
                r[e] = r_val
                p[e] = p_val

    # --- summary statistics over events ------------------------------------
    r_mean   = r.mean(axis=0)
    r_std    = r.std(axis=0)
    r_median = np.median(r, axis=0)
    outlier_threshold = r_mean - r_std

    # outlier_idx: events whose mean-r (averaged over any loop dims) is low
    r_event_mean = r.mean(axis=tuple(range(1, r.ndim))) if r.ndim > 1 else r
    outlier_idx  = np.where(r_event_mean < r_event_mean.mean() - r_event_mean.std())[0]

    summary_ = {
        'r_mean':            r_mean,
        'r_std':             r_std,
        'r_median':          r_median,
        'outlier_threshold': outlier_threshold,
        'outlier_idx':       outlier_idx,
        'n_outliers':        len(outlier_idx),
    }

    summary = AttrDict( summary_ )
    return r, p, summary


#################################################################################################
# vertical smoother
def smooth_z(f, z, wz, weight='dz'):
    """
    Smooth f(z) over a height window +/- wz, accounting for irregular
    vertical spacing in z.

    Parameters
    ----------
    f : np.ndarray, shape (n_z,)
    z : np.ndarray, shape (n_z,) — height in metres. Monotonic, but the
        direction doesn't matter (works for CAM top-down ordering,
        index 0 = top, index nz = surface, where z decreases with index).
    wz : float — half-width of the window in metres. At level i, averages
        over all j with |z[j]-z[i]| <= wz.
    weight : {'dz', 'none'}
        'dz'   : weight each level by its local z-spacing, so the result
                 approximates a height-weighted average rather than a
                 point-count average.
        'none' : simple unweighted mean of points in the window.

    Returns
    -------
    f_smooth : np.ndarray, shape (n_z,)
    """
    z = np.asarray(z, dtype=float)
    f = np.asarray(f, dtype=float)
    n = len(z)

    mask = np.abs(z[:, None] - z[None, :]) <= wz   # (n, n), row i = window around z[i]

    if weight == 'dz':
        dz = np.empty(n)
        dz[1:-1] = np.abs(z[2:] - z[:-2]) / 2
        dz[0]    = np.abs(z[1] - z[0])
        dz[-1]   = np.abs(z[-1] - z[-2])
        w = mask * dz[None, :]
    elif weight == 'none':
        w = mask.astype(float)
    else:
        raise ValueError("weight must be 'dz' or 'none'")

    return (w * f[None, :]).sum(axis=1) / w.sum(axis=1)


###############################################################################
# Big descriptive tile for plots
def big_title(Epl):
    safe_case = Epl.case.replace('_', r'\_')
    bold_case = rf"$\mathbf{{{safe_case}}}$"

    if type(Epl.threshold) is str:
        big_title = \
        f"N={Epl.N_events} of {Epl.Total_events} in {Epl.lon_range}X{Epl.lat_range}, \
        \n exclude orography={Epl.exclude_orography}, footprint={Epl.peak_footprint}, \
        \n epwp_thresh={Epl.threshold} at {Epl.zlev_event} km, \
        \n fraction of total epwp {Epl.Frac_of_total_epwp} ,\
        \n case = {bold_case} " 
    
    else:
        big_title = \
        f"N={Epl.N_events} of {Epl.Total_events} in {Epl.lon_range}X{Epl.lat_range}, \
        \n exclude orography={Epl.exclude_orography}, footprint={Epl.peak_footprint[0]}x{Epl.peak_footprint[1]}, \
        \n epwp_thresh={Epl.threshold[0]:.4g} to {Epl.threshold[1]:.2g} at {Epl.zlev_event/1000.:.2f} km, \
        \n fraction of total epwp {100*Epl.Frac_of_total_epwp:.2f}% ,\
        \n case = {bold_case} " 
    
    return big_title
    
###############################################################################
def plot_xavg_compos( El=None, fld=None, **kwargs ):

    if fld=="zeta_4D":
        fldlv=1.5e-5*np.linspace(-6,6,num=13)
        cmap='bwr'
    elif fld=='tilt_4D':
        fldlv=1.00e-7*np.linspace(0,6,num=31)
        cmap='coolwarm' #'inferno'
    elif fld=='fgf_4D':
        fldlv=1.5e-15*np.linspace(-6,6,num=31)
        cmap='bwr'
    elif fld=='thpwp_4D':
        fldlv=1.0*np.linspace(-0.05,0.05,num=31)
        cmap='bwr'
    else: 
        fldlv=31
        cmap='bwr'

    if "plot_v" in kwargs:
        plot_v_ = kwargs["plot_v"]
    else:
        plot_v_ = False


    
    ulv=np.linspace(-60,60,num=27)
    thlv=np.concatenate( (270.+np.arange(11)*10 , 380.+ np.arange(11)*20) )   #np.linspace(270,600,num=27)
    mflv=[0.001,0.002,.005, .01, .02]
    Epls=El #[El[0], El[1], El[2], El[3] ] #, Eco_0 ]
    print(thlv)
    nxplo,nyplo=len( Epls ),1
    fig,axs=plt.subplots( nyplo,nxplo , figsize=(nxplo*7+1,nyplo*8) )
    axs=axs.flatten()
    p=0
    
    for Epl in Epls:
        nv,nt_v,nz_v,ny_v,nx_v = np.shape( Epl.zeta_4D )
        zlev=Epl.zlevA
        delta_time=3
    
        ax=axs[p]
        Epl_vv=euti.avg_over_v(Epl)
        zeta_vxv=Epl_vv.zeta_4D.mean(axis=3)
        u_vxv=Epl_vv.u_4D.mean(axis=3)
        if plot_v_ == True:
            v_vxv=Epl_vv.v_4D.mean(axis=3)
        th_vxv=Epl_vv.th_4D.mean(axis=3)
        epwp_vxv=Epl_vv.epwp_4D.mean(axis=3)
        fld_vxv=Epl_vv[fld].mean(axis=3)
        colo = ax.contourf( np.arange(ny_v), zlev, fld_vxv[nt_v-1,:,:], cmap=cmap , levels=fldlv)
        lin1 = ax.contour( np.arange(ny_v), zlev, u_vxv[nt_v-1,:,:] , levels=ulv)
        ax.clabel(lin1, inline=True, fontsize=8, fmt='%1.0f')
        if plot_v_ == True:
            lin1b = ax.contour( np.arange(ny_v), zlev, v_vxv[nt_v-1,:,:] , levels=0.5*ulv, colors='blue')
            ax.clabel(lin1b, inline=True, fontsize=8, fmt='%1.0f')
        lin2 = ax.contour( np.arange(ny_v), zlev, th_vxv[nt_v-1,:,:] , levels=thlv, colors='red')
        ax.clabel(lin2, inline=True, fontsize=8, fmt='%1.0f')
        lin3 = ax.contour( np.arange(ny_v), zlev, epwp_vxv[nt_v-1,:,:] , levels=mflv, colors='black')
        ax.clabel(lin3, inline=True, fontsize=8, fmt='%1.3f')
        ax.set_ylim(0,20_000)
        ax.set_title( big_title(Epl) )
        p=p+1
    
    cax = fig.add_axes([0.15, 0.02, 0.70, 0.03])
    cbar = fig.colorbar(colo, cax=cax, orientation='horizontal')
    cbar.set_label(f"{fld}  s{r'$^{-1}$' }")

###############################################################################
def plot_yxavg_compos( El=None, **kwargs ):


    fields = [
        ('tilt_4D',     'tilt'),
        ('vmag_4D',     'wind speed'),
        ('abs_zeta_4D', 'absolute vorticity'),
    ]

    colors = ['red','blue','green']
    
    nxplo,nyplo=len( El ),1
    fig,axs=plt.subplots( nyplo,nxplo , figsize=(nxplo*7+1,nyplo*8) )
    axs=axs.flatten()

    p = 0
    
    for Epl in El:
        ax0 = axs[p]
        zlev = Epl.zlevA
        Epl_vv = euti.avg_over_v(Epl)
    
        # Three axes sharing the same y-axis
        ax1 = ax0.twiny()
        ax2 = ax0.twiny()
    
        # Move the third x-axis above the second one
        ax2.spines['top'].set_position(('outward', 40))
    
        xaxes = [ax0, ax1, ax2]

        f=0
        for axx, (fld, label) in zip(xaxes, fields):
            colorx=colors[f]
            fld_prof = np.mean(Epl_vv[fld], axis=(2, 3))
            line, = axx.plot(fld_prof[-1, :], zlev, lw=2.5, label=label,color=colorx)
    
            # Match each x-axis label and ticks to its curve
            color = line.get_color()
            axx.set_xlabel(label, color=color)
            axx.tick_params(axis='x', colors=color)
            axx.spines['top'].set_color(color)
            f +=1
        
        ax0.set_ylim(0, 20_000)
        ax0.set_ylabel('Height (m)')
        ax0.set_title(big_title(Epl), pad=55)
    
        p += 1
        
 
############################################################
# gw_05_steering_level.py
# ========================
# Profile-Based Steering Level Estimation for GW Source Analysis
# Part of the GW Source Analysis project (DYAMOND / 14km runs)
############################################################

def _trapz(y, x):
    """
    Local trapezoidal integration. Used instead of np.trapz/np.trapezoid
    to avoid depending on which name is available in a given numpy version.
    """
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))

############################################################
def compute_weighted_centroid_v2(w, z, z_min=None, z_max=None):
    """
    Vectorized, per-profile grids allowed. Exactly equivalent to the
    interpolate-cutoff + trapz version (same piecewise-linear integral).
    """
    w = np.asarray(w, dtype=float)
    z = np.asarray(z, dtype=float)
    if z.shape != w.shape:
        z = np.broadcast_to(z, w.shape)

    # make each profile ascending in z (flip descending rows)
    desc = z[..., -1] < z[..., 0]
    if np.any(desc):
        m = desc[..., None]
        z = np.where(m, z[..., ::-1], z)
        w = np.where(m, w[..., ::-1], w)

    z1, z2 = z[..., :-1], z[..., 1:]
    w1, w2 = w[..., :-1], w[..., 1:]

    # clip each segment to [z_min, z_max]
    a = np.maximum(z1, z_min) if z_min is not None else z1
    b = np.minimum(z2, z_max) if z_max is not None else z2

    dz    = z2 - z1
    valid = (b > a) & (dz > 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(dz > 0, (w2 - w1) / np.where(dz > 0, dz, 1.0), 0.0)

    wa = w1 + s * (a - z1)
    wb = w1 + s * (b - z1)
    zm = 0.5 * (a + b)
    wm = w1 + s * (zm - z1)

    seg_w  = np.where(valid, 0.5 * (wa + wb) * (b - a), 0.0)          # ∫ w dz
    seg_zw = np.where(valid,
                      (b - a) / 6.0 * (a*wa + 4.0*zm*wm + b*wb), 0.0) # ∫ z w dz (Simpson, exact)

    denom = seg_w.sum(axis=-1)
    numer = seg_zw.sum(axis=-1)

    out = np.full(denom.shape, np.nan)
    good = denom > 0
    out[good] = numer[good] / denom[good]
    return out
    
####################################################################################
def compute_weighted_centroid_fast(w, z, z_min=None, z_max=None):
    w = np.asarray(w, dtype=float)
    z = np.asarray(z, dtype=float)

    # collapse a tiled grid (identical rows) to 1D for the fast path
    if z.shape == w.shape and z.ndim > 1:
        z2 = z.reshape(-1, z.shape[-1])
        if (z2 == z2[0]).all():
            z = z2[0]

    if z.ndim != 1:
        return _compute_weighted_centroid_loop(w, z, z_min, z_max)  # old code

    batch_shape = w.shape[:-1]
    W = w.reshape(-1, w.shape[-1])

    order = np.argsort(z)
    zs = z[order]
    Ws = W[:, order]

    nan_out = np.full(W.shape[0], np.nan)

    if z_min is not None and z_min > zs[0]:
        if z_min >= zs[-1]:
            return nan_out.reshape(batch_shape)      # cutoff above profile top
        j = np.searchsorted(zs, z_min, side='right') # zs[j-1] <= z_min < zs[j]
        frac  = (z_min - zs[j-1]) / (zs[j] - zs[j-1])
        w_cut = Ws[:, j-1] + frac * (Ws[:, j] - Ws[:, j-1])
        zs = np.concatenate(([z_min], zs[j:]))
        Ws = np.concatenate([w_cut[:, None], Ws[:, j:]], axis=1)

    if z_max is not None and z_max < zs[-1]:
        if z_max <= zs[0]:
            return nan_out.reshape(batch_shape)      # cutoff below profile bottom
        j = np.searchsorted(zs, z_max, side='left')  # zs[j-1] < z_max <= zs[j]
        frac  = (z_max - zs[j-1]) / (zs[j] - zs[j-1])
        w_cut = Ws[:, j-1] + frac * (Ws[:, j] - Ws[:, j-1])
        zs = np.concatenate((zs[:j], [z_max]))
        Ws = np.concatenate([Ws[:, :j], w_cut[:, None]], axis=1)

    denom = np.trapz(Ws,      zs, axis=-1)
    numer = np.trapz(Ws * zs, zs, axis=-1)

    out = np.full(W.shape[0], np.nan)
    good = denom > 0
    out[good] = numer[good] / denom[good]
    return out.reshape(batch_shape)


####################################################################################
def compute_weighted_centroid(w, z, z_min=None, z_max=None):
    """
    Compute the height centroid of a positive-definite weight profile.

        z_centroid = integral(z * w, dz) / integral(w, dz)

    Physical motivation: the vertical analogue of using a vorticity-weighted
    horizontal centroid to define a storm's position. A profile concentrated
    near some level z0 is taken as a proxy for a balanced circulation
    centred near z0, so the steering wind should be sampled there rather
    than at an assumed fixed level.

    Parameters
    ----------
    w : np.ndarray, shape (..., n_z)
        Positive-definite weight profile(s), e.g. tilt, |zeta|, zeta**2.
        Leading dimensions are arbitrary batch dims (events, times, ...).
    z : np.ndarray, shape (n_z,) or same shape as w
        Height levels, irregular spacing allowed.
        If 1D, the same grid is used for every profile in w.
        If it matches w's shape, each profile may carry its own grid
        (e.g. z(t) for time-varying geopotential height).
    z_min : float or None
        If given, integration starts at this height; weight below z_min
        is excluded (e.g. to drop high surface-layer shear).
    z_max : float or None
        If given, integration ends at this height; weight above z_max
        is excluded (e.g. to drop upper-tropospheric/lower-stratospheric
        noise). Both cutoffs are obtained by linear interpolation, so
        neither needs to fall on a grid point.

    Returns
    -------
    z_centroid : np.ndarray, shape w.shape[:-1]
        Height centroid of each profile. NaN where the weight integral
        between z_min and z_max is zero (a cutoff excludes the whole
        profile, or the profile is all-zero).
    """
    w = np.asarray(w, dtype=float)
    z = np.asarray(z, dtype=float)

    if z.shape != w.shape:
        z = np.array(np.broadcast_to(z, w.shape))

    batch_shape = w.shape[:-1]
    n_batch     = int(np.prod(batch_shape)) if batch_shape else 1

    w_flat = w.reshape(n_batch, -1)
    z_flat = z.reshape(n_batch, -1)
    z_centroid = np.full(n_batch, np.nan)

    for i in range(n_batch):
        zi, wi = z_flat[i], w_flat[i]
        order  = np.argsort(zi)
        zi, wi = zi[order], wi[order]

        if z_min is not None and z_min > zi[0]:
            if z_min >= zi[-1]:
                continue  # cutoff at or above profile top -> no signal
            w_cut = np.interp(z_min, zi, wi)
            keep  = zi > z_min
            zi    = np.concatenate(([z_min], zi[keep]))
            wi    = np.concatenate(([w_cut], wi[keep]))

        if z_max is not None and z_max < zi[-1]:
            if z_max <= zi[0]:
                continue  # cutoff at or below profile bottom -> no signal
            w_cut = np.interp(z_max, zi, wi)
            keep  = zi < z_max
            zi    = np.concatenate((zi[keep], [z_max]))
            wi    = np.concatenate((wi[keep], [w_cut]))

        denom = _trapz(wi, zi)
        if denom > 0:
            z_centroid[i] = _trapz(zi * wi, zi) / denom

    return z_centroid.reshape(batch_shape)

####################################################################################

def _is_pressure_like(coord):
    """
    True if coordinate increases toward the surface (pressure-like),
    False if it decreases (height-like). Assumes CAM top-down layout
    (k=0 = TOA, k=pver-1 = surface); raises if columns disagree.
    """
    inc = coord[:, -1] > coord[:, 0]
    if inc.all():
        return True
    if (~inc).all():
        return False
    raise ValueError("mixed coordinate orientation across columns")

####################################################################################


def nearest_level(z_target, zm):
    """
    Map centroid heights to nearest level indices.

    z_target : (ncol,) centroid heights, NaN = no signal
    zm       : (ncol, pver) level heights, top-down (k=0 = TOA)

    Returns k : (ncol,) int, -1 sentinel where z_target is NaN.
    """
    k = np.full(z_target.shape, -1, dtype=int)
    valid = np.isfinite(z_target)
    if np.any(valid):
        d = np.abs(zm - z_target[..., None])
        k[valid] = np.argmin(d, axis=-1)[valid]
    return k

####################################################################################
def vorticity_centroid_levels(vorticity, coord, bound_surface=None, bound_top=None):
    """
    Steering level: centroid of |vorticity| between bound_surface and
                    bound_top. Launch level: centroid restricted to
                    levels above (physically higher than) the steering
                    level.

    vorticity : (ncol, pver)
    coord     : (ncol, pver) vertical coordinate, height (m) OR
                pressure (Pa), CAM top-down (k=0 = TOA)
    bound_surface : float or None, in coord units
        Near-surface cutoff (PBL exclusion), e.g. 1000. (m) or 85000. (Pa)
    bound_top : float or None, in coord units
        Upper cutoff, e.g. 20000. (m) or 5000. (Pa)

    Returns
    -------
    steering_level, launch_level : (ncol,) int, -1 = no signal
    c_steer, c_launch : (ncol,) float in coord units, NaN = no signal
    """
    absvort = np.abs(vorticity)
    ncol = absvort.shape[0]

    # map physical bounds to coordinate min/max
    if _is_pressure_like(coord):
        coord_min, coord_max = bound_top, bound_surface
    else:
        coord_min, coord_max = bound_surface, bound_top

    # --- first centroid: full column between the bounds ---
    c_steer = compute_weighted_centroid_v2(absvort, coord,
                                        z_min=coord_min, z_max=coord_max)
    steering_level = nearest_level(c_steer, coord)

    # --- restrict to levels above the steering level (smaller k = higher) ---
    w_top = absvort.copy()
    for i in range(ncol):
        k = steering_level[i]
        if k >= 0:
            w_top[i, k:] = 0.0
        else:
            w_top[i, :] = 0.0

    # --- second centroid: "top half" of the profile ---
    c_launch = compute_weighted_centroid_v2(w_top, coord)
    launch_level = nearest_level(c_launch, coord)

    return steering_level, launch_level, c_steer, c_launch
