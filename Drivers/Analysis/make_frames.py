#!/usr/bin/env python
"""
Render animation frames from MPAS-A 3.75 km DYAMOND output.

Regrids selected fields from the MPAS unstructured mesh to a regular lat/lon
grid using pre-computed bilinear weights, then writes one PNG per output time.

The colour image is one field at one level.  Two optional contour overlays are
available: relative vorticity (computed from U,V, at its OWN level, which is
normally lower than the plotted level) and precipitation.

Typical use
-----------
    # three test frames with the current settings
    ./make_frames.py --n-dates 3 --outdir frames_test --zeta --prec

    # full sequence
    ./make_frames.py --outdir frames_w --zeta --prec

    # one slice of the sequence (for a batch job array)
    ./make_frames.py --i0 0 --i1 25 --outdir frames_w --zeta --prec --skip-existing

Assemble afterwards with, e.g.:
    ffmpeg -framerate 12 -i frames_w/frame_%04d.png \
           -pix_fmt yuv420p -crf 18 anim.mp4
"""

import argparse
import os
import sys

# NOTE: the non-interactive "Agg" backend is selected inside main(), NOT here.
# Setting it at import time would break inline plotting in a notebook that
# imports this module.
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import ScalarFormatter

import numpy as np
import xarray as xr
from pathlib import Path

ROOTDIR = ".."
if ROOTDIR not in sys.path:
    sys.path.append(ROOTDIR)

from Utils import time_utils as tuti          # noqa: E402
from Utils import numerical_utils as nuti     # noqa: E402
import make_event_cubes as mec                # noqa: E402


# ----------------------------------------------------------------------
# Defaults (override on the command line)
# ----------------------------------------------------------------------
WGTFILE = "../mpasa3p75_TO_UHR_Lat_80S-0_Lon_0-360_bilin.nc"
CASEDIR = "/glade/campaign/cgd/amp/juliob/mpasa3p75km/cam77_dyamond1_prod1"
FTOPO = f"{CASEDIR}/TimeInvariant/PHIS_dyamond.nc"

ALL_FIELDS = ["U", "V", "U_prt", "V_prt", "PRECL", "theta_mpas", "w_mpas_prt"]
LEVELS = [5_000, 10_000, 15_000, 20_000]          # metres
TOPO_LEVELS = [1, 1000, 2000, 3000, 4000]

# kg m-2 s-1  ->  mm day-1
PREC_SCALE = 1_000.0 * 86_400.0


class AttrDict(dict):
    """dict that also supports attribute access: d.key as well as d['key']."""

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


# ----------------------------------------------------------------------
# Regridding
# ----------------------------------------------------------------------
def regrid(field_mpas, W, nlat, nlon):
    """
    field_mpas : np.ndarray, shape (n_src,) or (nz, n_src) or (nt, nz, n_src)
    Returns shape (..., nlat, nlon)
    """
    orig_shape = field_mpas.shape[:-1]
    flat = field_mpas.reshape(-1, field_mpas.shape[-1])   # (nfields, n_src)
    out = (W @ flat.T).T                                  # (nfields, nlat*nlon)
    return out.reshape(orig_shape + (nlat, nlon))


# ----------------------------------------------------------------------
# Date list
# ----------------------------------------------------------------------
def build_dates(start_date, nsteps, step_size):
    """Return a list of CAM-style date strings YYYY-MM-DD-SSSSS."""
    year, month, day, hour = start_date
    dates = []
    for _ in range(nsteps):
        dates.append(f"{year:04d}-{month:02d}-{day:02d}-{hour * 3_600:05d}")
        year, month, day, hour = tuti.increment_hours(
            [year, month, day, hour], nhours=step_size
        )
    return dates


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def resolve_file(datadir, fld, date_str):
    """Two naming conventions exist in this archive; pick whichever is present."""
    file1 = Path(f"{datadir}/{fld}_dyamond.nc")
    file2 = Path(f"{datadir}/{fld}_dyamond.{date_str}.nc")
    if file1.exists():
        return file1
    if file2.exists():
        return file2
    raise FileNotFoundError(f"neither {file1} nor {file2} exists")


def load_frame_data(date_str, fields, levels, W, nlat, nlon,
                    lat1d, lon1d, topo_yx, casedir, verbose=True):
    """Read and regrid all requested fields for one output date."""
    datadir = f"{casedir}/{date_str}"
    out = {
        "fields": fields, "levels": levels, "date": date_str,
        "lon": lon1d, "lat": lat1d, "topo": topo_yx,
    }

    for fld in fields:
        fname = resolve_file(datadir, fld, date_str)
        X = xr.open_dataset(fname)
        if verbose:
            print(f"  opened {fname}  dims={X[fld].dims}", flush=True)

        if "lev" in X[fld].dims:
            X = X.sel(lev=levels, method="nearest")
        elif "ilev" in X[fld].dims:
            X = X.sel(ilev=levels, method="nearest")

        fld_c = X[fld].values
        has_vertical = ("lev" in X[fld].dims) or ("ilev" in X[fld].dims)

        if has_vertical:
            nt, nz, _ = fld_c.shape
            fld_yx = np.zeros((nt, nz, nlat, nlon))
            for t in range(nt):
                for z in range(nz):
                    fld_yx[t, z, :, :] = regrid(fld_c[t, z, :], W, nlat, nlon)
        else:
            nt, _ = fld_c.shape
            fld_yx = np.zeros((nt, nlat, nlon))
            for t in range(nt):
                fld_yx[t, :, :] = regrid(fld_c[t, :], W, nlat, nlon)

        out[fld] = fld_yx
        X.close()

    return AttrDict(out)


def compute_zeta_2d(frame_data, it, iz):
    """
    Relative vorticity for ONE time and ONE level.

    Deliberately not looped over all levels: only the plotted slice is needed,
    and Sphere_Curl2 on a 3600x12410 grid is not cheap.
    """
    return nuti.Sphere_Curl2(
        f_x=frame_data.U[it, iz, :, :],
        f_y=frame_data.V[it, iz, :, :],
        lat=frame_data.lat, lon=frame_data.lon,
        wrap=True, verbose=False,
    )


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def coarsen2d(a, f):
    """Block-average a 2D array by factor f, trimming any remainder."""
    ny2 = (a.shape[0] // f) * f
    nx2 = (a.shape[1] // f) * f
    return a[:ny2, :nx2].reshape(ny2 // f, f, nx2 // f, f).mean(axis=(1, 3))


def coarsen1d(a, f):
    n2 = (a.shape[0] // f) * f
    return a[:n2].reshape(-1, f).mean(axis=1)


def make_topo_overlay(topo_yx, lat1d, lon1d, f):
    """Pre-compute the coarsened topography once; it is time-invariant."""
    return AttrDict({
        "topo": coarsen2d(topo_yx, f),
        "lat": coarsen1d(lat1d, f),
        "lon": coarsen1d(lon1d, f),
    })


def render_frame(data, lat1d, lon1d, topo_c, norm, outfile=None,
                 title=None, label="w (m/s)", cmap="bwr", scale=1.5, dpi=150,
                 cbar_ticks=None, cbar_format=None,
                 zeta=None, zeta_levels=None, zeta_coarsen=4,
                 zeta_color="g", zeta_lw=0.4, zeta_alpha=0.6,
                 prec=None, prec_levels=None, prec_coarsen=4,
                 prec_color="gray", prec_lw=0.4, prec_alpha=0.6,
                 xlim=None, ylim=None,
                 ):
    """
    Draw one frame.

    outfile given  -> write the PNG, close the figure, return None (batch use).
    outfile None   -> leave the figure open and return (fig, ax) so a notebook
                      can display it and keep tinkering.

    zeta, prec : optional 2D arrays on the same grid as `data`, drawn as
                 contour lines on top of the image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(scale * 16, scale * 5))

    im = ax.imshow(
        data, origin="lower",
        extent=[lon1d.min(), lon1d.max(), lat1d.min(), lat1d.max()],
        cmap=cmap, norm=norm, interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.6,
                        label=label, ticks=cbar_ticks, format=cbar_format)

    if zeta is not None:
        f = zeta_coarsen
        ax.contour(coarsen1d(lon1d, f), coarsen1d(lat1d, f), coarsen2d(zeta, f),
                   levels=zeta_levels, colors=zeta_color,
                   linewidths=zeta_lw, alpha=zeta_alpha)

    if prec is not None:
        f = prec_coarsen
        ax.contour(coarsen1d(lon1d, f), coarsen1d(lat1d, f), coarsen2d(prec, f),
                   levels=prec_levels, colors=prec_color,
                   linewidths=prec_lw, alpha=prec_alpha)

    ax.contour(topo_c.lon, topo_c.lat, topo_c.topo,
               levels=TOPO_LEVELS, colors="k", linewidths=0.5, alpha=0.7)

    ax.set_xlabel("Longitude (°E)", fontsize=10)
    ax.set_ylabel("Latitude (°N)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=12)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # NB: no bbox_inches='tight' -- every frame must have identical pixel
    # dimensions or ffmpeg will refuse the sequence.
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.10, top=0.93)

    if outfile is None:
        return fig, ax

    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def floats(s):
    """
    Parse a comma-separated list of floats, e.g. "-2e-4,2e-4".

    Given as ONE comma-separated string rather than several space-separated
    values because argparse mistakes bare negative values like -2e-4 for
    option flags.
    """
    if s is None or s.strip() == "":
        return None
    return [float(v) for v in s.split(",")]


def pair(s):
    """Parse "a,b" into a 2-tuple of floats, for axis limits."""
    if s is None or s.strip() == "":
        return None
    a, b = s.split(",")
    return (float(a), float(b))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Render MPAS DYAMOND animation frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- the colour image ---
    p.add_argument("--field", default="w_mpas_prt",
                   help="field rendered as the colour image")
    p.add_argument("--extra-fields", nargs="*", default=[],
                   help="additional fields to load (beyond --field)")
    p.add_argument("--levels", type=int, nargs="+", default=LEVELS,
                   help="target levels in metres, selected by nearest neighbour")
    p.add_argument("--lev-index", type=int, default=3,
                   help="index into --levels of the level to plot")
    p.add_argument("--time-index", type=int, default=0,
                   help="index into the time dim of each file")

    # --- colour mapping ---
    p.add_argument("--linthresh", type=float, default=0.75,
                   help="SymLogNorm linear threshold")
    p.add_argument("--vmax", type=float, default=2.0,
                   help="symmetric colour limit (+/-)")
    p.add_argument("--cmap", default="bwr")
    p.add_argument("--cbar-label", default="w (m/s)")
    p.add_argument("--cbar-ticks", type=floats,
                   default="-2,-1,-0.5,-0.2,0,0.2,0.5,1,2",
                   help="comma-separated colourbar tick values")

    # --- vorticity overlay ---
    p.add_argument("--zeta", action="store_true",
                   help="overlay relative vorticity contours (loads U and V)")
    p.add_argument("--zeta-lev-index", type=int, default=0,
                   help="index into --levels for the VORTICITY level; "
                        "independent of --lev-index, since the wave source is "
                        "normally well below the plotted wave field")
    p.add_argument("--zeta-levels", type=floats, default="-2e-4,2e-4",
                   help="comma-separated vorticity contour values (1/s)")
    p.add_argument("--zeta-color", default="g")
    p.add_argument("--zeta-alpha", type=float, default=0.25)
    p.add_argument("--zeta-coarsen", type=int, default=4)

    # --- precipitation overlay ---
    p.add_argument("--prec", action="store_true",
                   help="overlay precipitation contours (loads PRECL)")
    p.add_argument("--prec-field", default="PRECL")
    p.add_argument("--prec-levels", type=floats, default="3,25",
                   help="comma-separated precip contour values (mm/day)")
    p.add_argument("--prec-scale", type=float, default=PREC_SCALE,
                   help="multiplier converting the file units to mm/day")
    p.add_argument("--prec-color", default="gray")
    p.add_argument("--prec-alpha", type=float, default=0.2)
    p.add_argument("--prec-coarsen", type=int, default=4)

    # --- topography overlay ---
    p.add_argument("--coarsen", type=int, default=4,
                   help="block-averaging factor for the topography contours")

    # --- date sequence ---
    p.add_argument("--start-date", default="2016,8,1,3",
                   help="start as year,month,day,hour")
    p.add_argument("--nsteps", type=int, default=247)
    p.add_argument("--step-size", type=int, default=3, help="hours between frames")
    p.add_argument("--i0", type=int, default=0, help="first date index to render")
    p.add_argument("--i1", type=int, default=None, help="stop before this index")
    p.add_argument("--n-dates", type=int, default=None,
                   help="shorthand: render only the first N dates from --i0")

    # --- output ---
    p.add_argument("--outdir", default="frames")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--scale", type=float, default=1.5, help="figsize multiplier")
    p.add_argument("--xlim", type=pair, default=None,
                   help="longitude limits as 'lon0,lon1'")
    p.add_argument("--ylim", type=pair, default=None,
                   help="latitude limits as 'lat0,lat1'")
    p.add_argument("--skip-existing", action="store_true",
                   help="do not re-render frames that are already on disk")

    # --- paths ---
    p.add_argument("--wgtfile", default=WGTFILE)
    p.add_argument("--casedir", default=CASEDIR)
    p.add_argument("--ftopo", default=FTOPO)

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Batch rendering: draw straight to file buffers, never to a screen.
    matplotlib.use("Agg")

    # Load only what is actually needed.
    fields = [args.field] + [f for f in args.extra_fields if f != args.field]
    if args.zeta:
        for f in ("U", "V"):
            if f not in fields:
                fields.append(f)
    if args.prec and args.prec_field not in fields:
        fields.append(args.prec_field)

    os.makedirs(args.outdir, exist_ok=True)

    # --- one-time setup -------------------------------------------------
    print(f"reading weights from {args.wgtfile}", flush=True)
    W, nlat, nlon, lat1d, lon1d = mec.make_regridder(args.wgtfile)
    print(f"destination grid: nlat={nlat} nlon={nlon}", flush=True)

    with xr.open_dataset(args.ftopo) as Topo:
        htopo = Topo.PHIS.values / 9.8
    topo_yx = regrid(htopo, W, nlat, nlon)
    topo_c = make_topo_overlay(topo_yx, lat1d, lon1d, args.coarsen)

    # Fixed across all frames so the colours and colourbar do not flicker.
    norm = SymLogNorm(linthresh=args.linthresh, vmin=-args.vmax, vmax=args.vmax)

    start_date = [int(v) for v in args.start_date.split(",")]
    dates = build_dates(start_date, args.nsteps, args.step_size)

    i1 = args.i1 if args.i1 is not None else len(dates)
    if args.n_dates is not None:
        i1 = args.i0 + args.n_dates
    # Clamp: chunked/batch invocations routinely overshoot the end of the list.
    i1 = min(i1, len(dates))

    it = args.time_index
    iz = args.lev_index
    print(f"plotting {args.field} at {args.levels[iz]} m", flush=True)
    if args.zeta:
        print(f"vorticity overlay at {args.levels[args.zeta_lev_index]} m",
              flush=True)
    print(f"rendering dates [{args.i0}:{i1}] of {len(dates)}", flush=True)

    # --- frame loop -----------------------------------------------------
    for i in range(args.i0, i1):
        date_str = dates[i]
        outfile = os.path.join(args.outdir, f"frame_{i:04d}.png")

        if args.skip_existing and os.path.exists(outfile):
            print(f"[{i:04d}] {date_str}  exists, skipping", flush=True)
            continue

        print(f"[{i:04d}] {date_str}", flush=True)

        fd = load_frame_data(
            date_str, fields, args.levels, W, nlat, nlon,
            lat1d, lon1d, topo_yx, args.casedir,
        )

        fld = fd[args.field]
        data = fld[it, iz, :, :] if fld.ndim == 4 else fld[it, :, :]

        zeta2d = None
        if args.zeta:
            zeta2d = compute_zeta_2d(fd, it, args.zeta_lev_index)

        prec2d = None
        if args.prec:
            prec2d = args.prec_scale * fd[args.prec_field][it, :, :]

        render_frame(
            data, lat1d, lon1d, topo_c, norm, outfile,
            title=f"w at {args.levels[iz]} m:  {date_str}",
            label=args.cbar_label, cmap=args.cmap,
            scale=args.scale, dpi=args.dpi,
            cbar_ticks=args.cbar_ticks, cbar_format=ScalarFormatter(),
            zeta=zeta2d, zeta_levels=args.zeta_levels,
            zeta_coarsen=args.zeta_coarsen, zeta_color=args.zeta_color,
            zeta_alpha=args.zeta_alpha,
            prec=prec2d, prec_levels=args.prec_levels,
            prec_coarsen=args.prec_coarsen, prec_color=args.prec_color,
            prec_alpha=args.prec_alpha,
            xlim=args.xlim, ylim=args.ylim,
        )
        print(f"        wrote {outfile}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()