"""
Skill as a function of coarse-graining scale.

The point of the exercise is a parameterization: a CAM column at f09 asks "given
my resolved state, how much GW momentum flux should I launch?" So the target has
to be the flux for *one column*, and the score has to be measured there.

Three numbers float around this project and none of them are measured at that
scale:

  Julio's hand-built form   r = 0.578   sigma=2 smoothed f09
  our best PySR front       r ~ 0.70    sigma=2 smoothed f09
  Julio's event-composite NN r = 0.761  mean over an 11x11 f09 box (~10x14 deg)

The event windows turn out to sit on the same f09 grid as everything else
(dlon 1.25, dlat 0.9424), so an 11x11 window is ~1000 km across -- about 121 CAM
columns. And the gridded work inherits Julio's sigma=(0,2,2) Gaussian smoothing
from BasicAna4GWP-fitting.ipynb, so it is not a bare column either.

This script puts all of them on one axis: it re-derives the predictors and the
target at a ladder of smoothing scales from sigma=0 (a bare f09 column, what CAM
actually needs) out to an 11x11 box mean (Julio's event target), and fits the
same form-free model at each. The predictor set is the gridded counterpart of
his winning event set `tilt_levs+tilt+precl`:

    U_wv_src_mm      -> |V(z_launch) - V(z_steer)|
    avg(tilt) x4     -> tilt_zs, tilt_zl, tilt_zs_zl, tilt_zg_zs
    avg(precl)       -> precl

A gradient-boosted tree stands in for "any model of these inputs", so the curve
is a skill ceiling per scale, not one architecture's luck.

    conda activate npl-2026a && PYTHONNOUSERSITE=1 python smoothing_ladder.py
"""

import json

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.ensemble import HistGradientBoostingRegressor

ANA = "/glade/work/juliob/HiRes_ana_dev/Drivers/Analysis"
CASE = "cam77_dyamond1_prod1"
SRC = f"{ANA}/{CASE}_src_fit_vars_4.nc"
TOPO = ("/glade/work/juliob/Topo/NCARTopoJTB/cases/fv1x1_Sco100_GrnlAnt/output/"
        "fv1x1_gmted2010_modis_bedmachine_nc3000_Laplace0100_noleak_"
        "greenlndantarcsgh30fac2.50_20251009.nc")
OUT = "/glade/work/wchapman/PYSR_Julio/smoothing_ladder.json"

FIELDS = ["epwp_zl", "tilt_zs", "tilt_zl", "tilt_zs_zl", "tilt_zg_zs",
          "precl", "u_zs", "v_zs", "u_zl", "v_zl"]

# (label, kind, width). sigma=2 is what the rest of this directory uses;
# box=11 reproduces the event-composite target.
LADDER = [
    ("bare f09 column", "gauss", 0.0),
    ("sigma=1", "gauss", 1.0),
    ("sigma=2  (our gridded work)", "gauss", 2.0),
    ("sigma=3", "gauss", 3.0),
    ("sigma=5", "gauss", 5.0),
    ("5x5 box", "box", 5),
    ("11x11 box  (Julio's event target)", "box", 11),
]

P_FLOOR = 1e-4 * 3e-8        # matches run_pysr_gwsrc.py's floor on precl
TRAIN_FRAC = 0.7
MAX_ROWS = 400_000           # subsample for tree-fitting speed
SEED = 0


def skill(pred, true):
    """Log-space r, R2 and calibration slope, in float64."""
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    r = float(np.corrcoef(p, t)[0, 1])
    denom = float(np.sum((t - t.mean()) ** 2))
    r2 = float(1.0 - np.sum((p - t) ** 2) / denom)
    pm, tm = p.mean(), t.mean()
    slope = float(np.sum((p - pm) * (t - tm)) / np.sum((p - pm) ** 2))
    return dict(r=r, r2=r2, slope=slope)


def coarse(q, kind, width):
    """Horizontal coarse-graining, periodic in longitude, reflecting in lat."""
    if kind == "gauss":
        if width == 0:
            return q
        return gaussian_filter(q, sigma=(0, width, width),
                               mode=("reflect", "reflect", "wrap"))
    return uniform_filter(q, size=(1, width, width),
                          mode=("reflect", "reflect", "wrap"))


def main():
    ds = xr.open_dataset(SRC)
    lat, nx = ds.lat.values, ds.sizes["lon"]
    raw = {f: ds[f].values.astype(np.float64) for f in FIELDS}

    mxd = gaussian_filter(xr.open_dataset(TOPO).MXDIS.values[0], sigma=2)
    latyx = np.tile(lat[:, None], (1, nx))
    masks = {
        "sh": (mxd < 200.) & (latyx >= -80.) & (latyx <= -20.),
        "nh": (mxd < 200.) & (latyx <= 80.) & (latyx >= 0.),
    }

    nt = raw["epwp_zl"].shape[0]
    t_split = int(TRAIN_FRAC * nt)
    rng = np.random.default_rng(SEED)
    results = []

    for label, kind, width in LADDER:
        sm = {f: coarse(q, kind, width) for f, q in raw.items()}

        data = {}
        for tag, mask in masks.items():
            col = {f: q[:, mask] for f, q in sm.items()}
            y = col["epwp_zl"]
            ok = y > 0
            # |V(launch) - V(steer)|, the gridded twin of U_wv_src_mm
            usrc = np.sqrt((col["u_zl"] - col["u_zs"]) ** 2
                           + (col["v_zl"] - col["v_zs"]) ** 2)
            X = np.stack([usrc, col["tilt_zs"], col["tilt_zl"],
                          col["tilt_zs_zl"], col["tilt_zg_zs"],
                          np.maximum(col["precl"], P_FLOOR)], axis=-1)
            tidx = np.repeat(np.arange(nt), mask.sum()).reshape(y.shape)
            data[tag] = (X[ok], np.log(y[ok]), tidx[ok])

        Xs, ys, ts = data["sh"]
        tr, te = ts < t_split, ts >= t_split

        def sub(m):
            idx = np.flatnonzero(m)
            if idx.size > MAX_ROWS:
                idx = rng.choice(idx, MAX_ROWS, replace=False)
            return idx

        itr = sub(tr)
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
            early_stopping=True, random_state=SEED)
        model.fit(Xs[itr], ys[itr])

        ite = sub(te)
        Xn, yn, _ = data["nh"]
        inh = sub(np.ones(yn.size, bool))

        rec = {
            "label": label, "kind": kind, "width": float(width),
            "var_log_target_sh": float(ys.var()),
            "n_sh": int(ys.size),
            "test": skill(model.predict(Xs[ite]), ys[ite]),
            "nh": skill(model.predict(Xn[inh]), yn[inh]),
        }
        results.append(rec)
        print(f"{label:<36s} var={rec['var_log_target_sh']:.3f}  "
              f"SH-test r={rec['test']['r']:.3f} R2={rec['test']['r2']:+.3f}  "
              f"NH r={rec['nh']['r']:.3f} slope={rec['nh']['slope']:.2f}",
              flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
