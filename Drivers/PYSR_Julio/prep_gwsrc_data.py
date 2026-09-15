"""
Extract and cache the GW-source fitting dataset used by pysr_gwsrc_fit.ipynb.

Reproduces the preprocessing in Julio's BasicAna4GWP-fitting.ipynb (cells 19-20,
33) and writes a compact .npz so the PySR notebook needs neither netCDF4 nor the
845 MB source file.

Julio's pipeline, followed exactly for the baseline variables:
  - fields from {case}_src_fit_vars_4.nc, shape (time=247, lat=192, lon=288)
  - horizontal Gaussian smoothing, sigma=(0,2,2), mode=(reflect,reflect,wrap)
  - mask on smoothed MXDIS < 200 m (drop orographic-GW columns) and a lat band
  - SH band [-80,-20] is the fitting hemisphere; NH [0,80] is the transfer test

Run in an env that has netCDF4 (the PySR env does not):
    conda activate npl-2026a && PYTHONNOUSERSITE=1 python prep_gwsrc_data.py
"""

import os

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

ANA = "/glade/work/juliob/HiRes_ana_dev/Drivers/Analysis"
CASE = "cam77_dyamond1_prod1"
SRC = f"{ANA}/{CASE}_src_fit_vars_4.nc"
TOPO = ("/glade/work/juliob/Topo/NCARTopoJTB/cases/fv1x1_Sco100_GrnlAnt/output/"
        "fv1x1_gmted2010_modis_bedmachine_nc3000_Laplace0100_noleak_"
        "greenlndantarcsgh30fac2.50_20251009.nc")
# Horizontal smoothing width, in f09 grid cells (dlon 1.25, dlat 0.9424).
# **Default 2**, matching Julio's notebook: each sample becomes a Gaussian-
# weighted mean of ~50 neighbouring columns, ~180 x 210 km at 50 deg. That is a
# real choice, not a formatting detail -- it sets the scale the fitted model is a
# parameterization *for*, and skill moves ~0.14 of r across the ladder
# (smoothing_ladder.py). It is kept because the curve flattens here: 11x11 buys
# +0.013 SH inside the noise, and the NH degrades beyond sigma=2. GWSRC_SIGMA=0
# gives bare columns, the reference for a strictly per-column source function.
# The cache filename always carries the scale, because a score is meaningless
# without it.
SIGMA = float(os.environ.get("GWSRC_SIGMA", 2))
OUT = f"/glade/work/wchapman/PYSR_Julio/gwsrc_data_s{SIGMA:g}.npz"

# Fields carried straight through the smoothing. epwp_zl is the target; the rest
# are candidate predictors. Julio's fit used only tilt_zs and precl.
FIELDS = ["epwp_zl", "tilt_zs", "tilt_zl", "tilt_zs_zl", "tilt_zg_zs",
          "zeta_zs", "zeta_zl", "precl", "u_zs", "v_zs", "u_zl", "v_zl"]

# Level indices are integers; smoothing them horizontally is still sensible
# (they index a monotone pressure axis) but they are kept separate for clarity.
KFIELDS = ["k_launch", "k_steer"]


def main():
    ds = xr.open_dataset(SRC)
    lat, lon, lev = ds.lat.values, ds.lon.values, ds.lev.values
    nt, ny, nx = ds.epwp_zl.shape
    print(f"source grid: nt={nt} ny={ny} nx={nx}")

    topo = xr.open_dataset(TOPO)
    mxdis = topo.MXDIS.values[0, :, :]
    mxd_smooth = gaussian_filter(mxdis, sigma=2)

    latyx = np.tile(lat[:, None], (1, nx))
    mask_sh = (mxd_smooth < 200.) & (latyx >= -80.) & (latyx <= -20.)
    mask_nh = (mxd_smooth < 200.) & (latyx <= 80.) & (latyx >= 0.)
    print(f"SH mask columns: {mask_sh.sum()}  -> {nt * mask_sh.sum():,} samples")
    print(f"NH mask columns: {mask_nh.sum()}  -> {nt * mask_nh.sum():,} samples")

    # Smooth every field once, then harvest both hemispheres from it.
    print(f"horizontal smoothing: sigma={SIGMA} grid cells "
          f"({'none -- bare f09 columns' if SIGMA == 0 else 'Gaussian'})")
    smoothed = {}
    for name in FIELDS:
        q = ds[name].values.astype(np.float64)
        smoothed[name] = q if SIGMA == 0 else gaussian_filter(
            q, sigma=(0, SIGMA, SIGMA), mode=("reflect", "reflect", "wrap"))
        print(f"  smoothed {name}")

    # Steering/launch level indices -> altitude. `lev` here is height in metres,
    # descending from the model top (41116 m) to the surface (35 m), so the
    # index maps straight onto it with no pressure conversion.
    z_of_lev = np.asarray(lev, float)
    for name in KFIELDS:
        k = ds[name].values
        smoothed["z_" + name.split("_")[1]] = z_of_lev[np.clip(k, 0, len(lev) - 1)]
    # Steering level sits *below* launch level (medians 6.5 km vs 10.0 km), so
    # take launch-minus-steer to keep this depth positive.
    smoothed["dz_launch_steer"] = smoothed["z_launch"] - smoothed["z_steer"]

    # Time index per sample, so the notebook can split on contiguous time blocks
    # rather than at random: neighbouring times are strongly autocorrelated and a
    # random split would leak.
    tidx = np.arange(nt, dtype=np.int16)

    out = {"lev": lev, "lat": lat, "lon": lon}
    for tag, mask in (("sh", mask_sh), ("nh", mask_nh)):
        ncol = int(mask.sum())
        out[f"tidx_{tag}"] = np.repeat(tidx, ncol)
        for name, q in smoothed.items():
            out[f"{name}_{tag}"] = q[:, mask].ravel().astype(np.float32)
        print(f"{tag}: {out[f'tidx_{tag}'].size:,} samples")

    # Write-then-rename, so a rerun cannot hand a half-written archive to a
    # search job that reopens the cache mid-flight. os.replace is atomic within
    # a filesystem and leaves already-open handles on the old inode.
    tmp = OUT + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
