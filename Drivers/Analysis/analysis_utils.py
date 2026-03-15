import numpy as np
from scipy.ndimage import label
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


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

    mask = epwp > thresh

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


def find_gw_events_watershed(epwp, thresh):

    mask = epwp > thresh

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

    return events

def composite4D( event_list, aa , lat, lon, window=[5,5,5] , TZHkey='tzyx', lat_range=[-90,90], lon_range=[0,360] ):
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
    
    print( count )

    wt,wy,wx = window
    if TZHkey == 'tzyx':
        aa_pad = np.pad(aa, ((wt, wt), (0, 0), (wy, wy), (0, 0)), mode='edge')
        aa_pad = np.pad(aa_pad, ((0, 0), (0, 0), (0, 0), (wx, wx)), mode='wrap')
        aa_comp = np.zeros( ( count , wt+1, nz, 2*wy+1, 2*wx+1 )  )
    elif TZHkey == 'tyx':
        aa_pad = np.pad(aa, ((wt, wt), (wy, wy), (0, 0)), mode='edge')
        aa_pad = np.pad(aa_pad, ((0, 0), (0, 0), (wx, wx)), mode='wrap')
        aa_comp = np.zeros( ( count , wt+1, 2*wy+1, 2*wx+1 )  )
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
                    aa_comp[c, 0:wt+1, :,0:2*wy+1, 0:2*wx+1 ] = aa_pad[t_-wt:t_+1, :, y-wy:y+wy+1, x-wx:x+wx+1 ] 
                elif TZHkey == 'tyx':
                    aa_comp[c, 0:wt+1, 0:2*wy+1, 0:2*wx+1 ] = aa_pad[t_-wt:t_+1 ,y-wy:y+wy+1,x-wx:x+wx+1]                 
                c=c+1
    
    print( c )
    
    
    #for t in np.arange( nt ):
    #    for ie in np.arange( nevs ):
    return aa_comp



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





