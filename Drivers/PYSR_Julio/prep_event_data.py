"""
Stage 1 of the Julio-NN skill analysis: turn the two event-composite pickles
into a small .npz of predictors and targets.

Julio's message (2026-08-17) named the current pair and the winning predictor
set. The pickles are 65 GB (SH) and 26 GB (NH) and are written by numpy 2, so
they can only be opened in an env with numpy >= 2 -- which here means npl-2026a,
which has no torch. Hence the split: extract here, fit in nn_event_gwsrc.py.

    conda activate npl-2026a
    PYTHONNOUSERSITE=1 python prep_event_data.py

Everything through the predictor build is a verbatim port of Julio's pipeline
(see julio_event_vendor.py), including the sweep3 hyperparameters:
predictorSet='Dynamic-steer-launch', z_targ=-1, yrange=[1,-1], time-averaged
events, and precl converted to mm/day before it is ever used.
"""

import os
import pickle
import sys
import time

import numpy as np

import julio_event_vendor as V

BDIR = "/glade/derecho/scratch/juliob/archive/GW_event_analysis/PKL"
STEM = "cam77_dyamond1_prod1_2016-08-01-10800-x-2016-08-31-75600"
SUFF = "ocean_EvZ10km_rho_epwp_subt_ftp5X5_Frac100%_v4.pkl"

# NH first: it is the smaller file, so a bug in the predictor build surfaces
# after a 26 GB read rather than a 65 GB one.
FILES = {
    "nh": f"{BDIR}/{STEM}_35N-70N_{SUFF}",
    "sh": f"{BDIR}/{STEM}_60S-40S_{SUFF}",
}

OUT = "/glade/work/wchapman/PYSR_Julio/event_data.npz"

# sweep3's five sets. The first field of each list is the level-finder, so all
# five share the same steering/launch levels; only the flux predictors differ.
PREDICTOR_SETS = [
    ("tilt_levs+tilt", ["tilt_4D", "tilt_4D"]),
    ("tilt_levs+precl", ["tilt_4D", "precl_4D"]),
    ("tilt_levs+tilt+precl", ["tilt_4D", "tilt_4D", "precl_4D"]),  # Julio's best
    ("tilt_levs+fgf", ["tilt_4D", "fgf_4D"]),
    ("tilt_levs+tilt+fgf", ["tilt_4D", "tilt_4D", "fgf_4D"]),
]

YRANGE = [1, -1]
SCALE_PRECIP = 86_400.0 * 1_000.0   # kg m-2 s-1 -> mm/day


def load_hemisphere(path):
    """Load El from a pickle and return its first (lowest) threshold element,
    time-averaged and with precl in mm/day."""
    print(f"reading {os.path.basename(path)} "
          f"({os.path.getsize(path) / 2**30:.1f} GiB) ...", flush=True)
    t0 = time.time()
    with open(path, "rb") as f:
        El = pickle.load(f)
    print(f"  loaded {len(El)} threshold elements in {time.time() - t0:.0f} s",
          flush=True)

    Eco = El[0]
    del El                      # 60+ GB of higher thresholds we never use

    print(f"  case={Eco.case}  lat_range={Eco.lat_range}  "
          f"N_events={Eco.get('N_events', '?')}  "
          f"peak_footprint={Eco.peak_footprint}", flush=True)
    for k in ("u_4D", "tilt_4D", "precl_4D", "epwp_4D", "zeta_4D", "fgf_4D"):
        if k in Eco:
            print(f"  {k}: {np.shape(Eco[k])}")

    Eco = V.vtime_avg_events(Eco)
    Eco.precl_4D = SCALE_PRECIP * Eco.precl_4D
    return Eco


def main():
    V.install_event_utils_stub()

    out = {}
    for tag, path in FILES.items():
        Eco = load_hemisphere(path)
        zlev = np.asarray(Eco.zlevA, float)

        zs = zl = None
        for name, use_pred in PREDICTOR_SETS:
            t0 = time.time()
            preds, pnames, _, key_z, _, zs, zl = V.dynamic_steer_launch(
                Eco=Eco, zlev=zlev, use_predictors=use_pred,
                trange=None, yrange=YRANGE, xrange=None, verbose=False)
            X = np.column_stack(preds).astype(np.float32)
            out[f"X_{tag}_{name}"] = X
            out[f"names_{tag}_{name}"] = np.array(pnames)
            print(f"  [{tag}] {name}: X={X.shape}  ({time.time() - t0:.0f} s)",
                  flush=True)

        # Target and the footprint it is averaged over. All five sets share the
        # same level-finder, so one target serves them all.
        yv = V.build_target(Eco, zl_dyn=zl)
        out[f"y_{tag}"] = yv.astype(np.float64)
        out[f"tidx_{tag}"] = np.asarray(Eco.time4D)
        out[f"lat_{tag}"] = np.asarray(Eco.lat4D, np.float32)
        out[f"lon_{tag}"] = np.asarray(Eco.lon4D, np.float32)
        out[f"zs_{tag}"] = zs[:, -1].astype(np.int16)
        out[f"zl_{tag}"] = zl[:, -1].astype(np.int16)
        out[f"zlev_{tag}"] = zlev

        # The raw footprint the target averages over. This is the quantity that
        # decides whether the event framework's higher skill is a better model
        # or an easier target -- see the ceiling analysis in the notebook.
        nv = yv.size
        foot = np.empty((nv,) + np.shape(Eco.epwp_4D)[3:], np.float32)
        for v in range(nv):
            foot[v] = Eco.epwp_4D[v, -1, zl[v, -1], :, :]
        out[f"foot_{tag}"] = foot
        print(f"  [{tag}] target: n={nv}  footprint={foot.shape[1:]}  "
              f"median={np.median(yv):.4g}", flush=True)

        del Eco, foot

    tmp = OUT + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, OUT)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 2**20:.1f} MiB)")


if __name__ == "__main__":
    sys.exit(main())
