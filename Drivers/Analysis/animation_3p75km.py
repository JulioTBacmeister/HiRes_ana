#!/usr/bin/env python
"""
Render animation frames from MPAS-A 3.75 km DYAMOND output.

Regrids selected fields from the MPAS unstructured mesh to a regular lat/lon
grid using pre-computed bilinear weights, then writes one PNG per output time.

Typical use
-----------
    # quick look at the first 3 dates
    ./make_frames.py --n-dates 3 --outdir frames_test

    # full sequence
    ./make_frames.py --outdir frames_w

    # one slice of the sequence (for a batch job array)
    ./make_frames.py --i0 0 --i1 60 --outdir frames_w --skip-existing

Assemble afterwards with, e.g.:
    ffmpeg -framerate 12 -i frames_w/frame_%04d.png \
           -pix_fmt yuv420p -crf 18 anim.mp4
"""

import argparse
import os
import sys

# Non-interactive backend: must be set before pyplot is imported.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

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
LEVELS = [5_000, 10_000, 15_000, 20_000]
TOPO_LEVELS = [1, 1000, 2000, 3000, 4000]


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


def compute_zeta(frame_data, verbose=True):
    """Relative vorticity from the regridded U, V."""
    nt, nz, ny, nx = frame_data.U.shape
    zeta = np.zeros((nt, nz, ny, nx))
    for t in range(nt):
        for z in range(nz):
            zeta[t, z, :, :] = nuti.Sphere_Curl2(
                f_x=frame_data.U[t, z, :, :],
                f_y=frame_data.V[t, z, :, :],
                lat=frame_data.lat, lon=frame_data.lon,
                wrap=True, verbose=False,
            )
        if verbose:
            print(f"  zeta: t={t} done", flush=True)
    return zeta


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


def render_frame(data, lat1d, lon1d, topo_c, norm, outfile,
                 title=None, label="w (m/s)", cmap="bwr", scale=1.5, dpi=150):
    """Draw one frame and write it to disk."""
    fig, ax = plt.subplots(1, 1, figsize=(scale * 16, scale * 5))

    im = ax.imshow(
        data, origin="lower",
        extent=[lon1d.min(), lon1d.max(), lat1d.min(), lat1d.max()],
        cmap=cmap, norm=norm, interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.6, label=label)

    ax.contour(topo_c.lon, topo_c.lat, topo_c.topo,
               levels=TOPO_LEVELS, colors="k", linewidths=0.5, alpha=0.7)

    ax.set_xlabel("Longitude (°E)", fontsize=10)
    ax.set_ylabel("Latitude (°N)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=12)

    # NB: no bbox_inches='tight' -- every frame must have identical pixel
    # dimensions or ffmpeg will refuse the sequence.
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.10, top=0.93)
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Render MPAS DYAMOND animation frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # what to plot
    p.add_argument("--field", default="w_mpas_prt",
                   help="field to render as the color image")
    p.add_argument("--extra-fields", nargs="*", default=[],
                   help="additional fields to load (beyond --field)")
    p.add_argument("--lev-index", type=int, default=3,
                   help="index into --levels of the level to plot")
    p.add_argument("--time-index", type=int, default=0,
                   help="index into the time dim of each file")
    p.add_argument("--levels", type=int, nargs="+", default=LEVELS,
                   help="target levels (Pa) selected by nearest neighbour")
    p.add_argument("--compute-zeta", action="store_true",
                   help="compute relative vorticity (forces loading U and V)")

    # date sequence
    p.add_argument("--start-date", default="2016,8,1,3",
                   help="start as year,month,day,hour")
    p.add_argument("--nsteps", type=int, default=247)
    p.add_argument("--step-size", type=int, default=3, help="hours between frames")
    p.add_argument("--i0", type=int, default=0, help="first date index to render")
    p.add_argument("--i1", type=int, default=None, help="stop before this index")
    p.add_argument("--n-dates", type=int, default=None,
                   help="shorthand: render only the first N dates")

    # color mapping
    p.add_argument("--linthresh", type=float, default=1.0,
                   help="SymLogNorm linear threshold")
    p.add_argument("--vmax", type=float, default=5.0,
                   help="symmetric color limit (+/-)")
    p.add_argument("--cmap", default="bwr")
    p.add_argument("--cbar-label", default=None)

    # output
    p.add_argument("--outdir", default="frames")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--scale", type=float, default=1.5, help="figsize multiplier")
    p.add_argument("--coarsen", type=int, default=4,
                   help="block-averaging factor for the topography contours")
    p.add_argument("--skip-existing", action="store_true",
                   help="do not re-render frames that are already on disk")

    # paths
    p.add_argument("--wgtfile", default=WGTFILE)
    p.add_argument("--casedir", default=CASEDIR)
    p.add_argument("--ftopo", default=FTOPO)

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Load only what is actually needed.
    fields = [args.field] + [f for f in args.extra_fields if f != args.field]
    if args.compute_zeta:
        for f in ("U", "V"):
            if f not in fields:
                fields.append(f)

    os.makedirs(args.outdir, exist_ok=True)

    # --- one-time setup -------------------------------------------------
    print(f"reading weights from {args.wgtfile}", flush=True)
    W, nlat, nlon, lat1d, lon1d = mec.make_regridder(args.wgtfile)
    print(f"destination grid: nlat={nlat} nlon={nlon}", flush=True)

    with xr.open_dataset(args.ftopo) as Topo:
        htopo = Topo.PHIS.values / 9.8
    topo_yx = regrid(htopo, W, nlat, nlon)
    topo_c = make_topo_overlay(topo_yx, lat1d, lon1d, args.coarsen)

    # Fixed across all frames so the colors and colorbar do not flicker.
    norm = SymLogNorm(linthresh=args.linthresh, vmin=-args.vmax, vmax=args.vmax)
    cbar_label = args.cbar_label or args.field

    start_date = [int(v) for v in args.start_date.split(",")]
    dates = build_dates(start_date, args.nsteps, args.step_size)

    i1 = args.i1 if args.i1 is not None else len(dates)
    if args.n_dates is not None:
        i1 = min(args.i0 + args.n_dates, len(dates))

    print(f"rendering dates [{args.i0}:{i1}] of {len(dates)}", flush=True)

    # --- frame loop -----------------------------------------------------
    for i in range(args.i0, i1):
        date_str = dates[i]
        outfile = os.path.join(args.outdir, f"frame_{i:04d}.png")

        if args.skip_existing and os.path.exists(outfile):
            print(f"[{i:04d}] {date_str}  exists, skipping", flush=True)
            continue

        print(f"[{i:04d}] {date_str}", flush=True)

        frame_data = load_frame_data(
            date_str, fields, args.levels, W, nlat, nlon,
            lat1d, lon1d, topo_yx, args.casedir,
        )

        if args.compute_zeta:
            frame_data["ZETA"] = compute_zeta(frame_data)

        fld = frame_data[args.field]
        if fld.ndim == 4:
            data = fld[args.time_index, args.lev_index, :, :]
        else:
            data = fld[args.time_index, :, :]

        render_frame(
            data, lat1d, lon1d, topo_c, norm, outfile,
            title=f"{args.field}   {date_str}",
            label=cbar_label, cmap=args.cmap,
            scale=args.scale, dpi=args.dpi,
        )
        print(f"        wrote {outfile}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()