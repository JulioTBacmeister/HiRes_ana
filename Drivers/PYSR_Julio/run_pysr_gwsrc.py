"""
Symbolic regression for the gravity-wave source function, replacing the
hand-built form in Julio's BasicAna4GWP-fitting.ipynb:

    tau = y0 + A (D/D_REF)^a / (1 + D/D0) + B (P/P_REF)^b (D/D_REF)^c

which scores r = 0.5784 in log space on the SH sample.

Three feature sets, selected by the first command-line argument:

  dp    only D = tilt_zs and P = precl, the predictors Julio used. A control:
        the binned variance decomposition caps *any* f(D, P) at r ~= 0.583, so
        this run should land near 0.578 and go no further. It does -- a full
        hour on 32 cores with 25-node expressions reaches 0.584 in-sample and
        0.574 held-out, against the hand-built form's 0.578. The form was never
        the bottleneck; the inputs are.

  tilt  the tilt family plus precl. The conservative set: it improves on the
        baseline whether fitted to one hemisphere or both.

  full  eight scaled predictors including wind speeds and steering height. The
        most skilful set, but only when fitted globally -- see PYSR_TRAIN.

Fitting is done on log tau, not tau. The score Julio reports is a correlation
in log space, so this optimises the metric directly, and it conditions a target
spanning 4.5 decades into an O(1)-O(10) range -- the single most useful thing
you can do for a symbolic regression (see pysrtips_bety260326.py).

Environment:
    PYSR_TRAIN    'sh' (default) or 'both'. See the note above FEATURE_SETS --
                  this changes which predictors are worth having, not just the
                  scores. With 'both', the r_nh column is held-out *times* in
                  the NH; with 'sh' it is a full transfer test. Different
                  questions, same column name.
    PYSR_TIMEOUT  wall-clock bound on the search, seconds. The real budget is
                  niterations * populations, which overruns any walltime.
    RUN_TAG       suffix for the run directory and front file, so concurrent
                  searches of one mode do not overwrite each other.

Usage:
    conda activate /glade/work/wchapman/conda-envs/L96M2lines
    python run_pysr_gwsrc.py {dp|tilt|full} [niterations]
"""

import os
import sys
import json
import time

import numpy as np
from pysr import PySRRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
# Which coarse-graining the fit targets, in f09 cells. Default 2 -- Julio's
# smoothing, ~50 columns averaged; the skill-vs-scale curve flattens there, so
# coarser buys nothing (smoothing_ladder.py). Must match a cache built by
# prep_gwsrc_data.py at the same GWSRC_SIGMA; a score is only comparable to
# other scores measured at the same scale.
SIGMA = float(os.environ.get("GWSRC_SIGMA", 2))
DATA = os.path.join(HERE, f"gwsrc_data_s{SIGMA:g}.npz")
OUTDIR = os.path.join(HERE, "pysr_outputs")

# Reference values. D_REF and P_REF match source_fit_refvals.py so the fitted
# coefficients stay comparable with Julio's; the rest are set near the median of
# their variable so every column lands in O(1)-O(10).
D_REF = 3e-7        # tilt
P_REF = 3e-8        # precl
Y_REF = 4.68e-3     # epwp_zl, median of the SH sample
U_REF = 10.0        # wind speed, m/s
Z_REF = 5000.0      # height, m
Zeta_REF = 3e-5     # relative vorticity, 1/s

# precl runs down to 1e-19 of P_REF in nominally dry columns. Left alone, any
# log(P) or 1/P subexpression detonates on a single dry point and takes the
# whole batch with it. Flooring at 1e-4 of P_REF touches ~1.2% of the sample and
# still lets P^b vanish for dry columns, which is the behaviour the additive
# form was built to have.
P_FLOOR = 1e-4

# Which predictors each mode searches over. Which set is *best* depends on what
# you train on, and the two answers disagree (see nn_gwsrc.py):
#
#   trained on SH only -- the wind and height columns in `full` learn
#   SH-specific structure. NH R2 goes negative, and even after an optimal affine
#   recalibration they transfer no better than plain (D, P) (NH R2cal ~0.33-0.35
#   against 0.35). `tilt` is the only set that clearly transfers, NH R2cal 0.50.
#
#   trained on both hemispheres -- the same wind columns become the best
#   predictors available, lifting NH r from 0.652 (D,P) to 0.743, with a
#   calibration slope near 0.92. Nothing is left chasing one hemisphere's
#   weather.
#
# So `full` + PYSR_TRAIN=both is the configuration to take seriously, and `tilt`
# is the conservative fallback that works either way.
FEATURE_SETS = {
    "dp":   ["D", "P"],
    "tilt": ["D", "Dl", "Dg", "Dsl", "P"],
    "full": ["D", "Dl", "Dg", "P", "Us", "Ul", "Zs", "Zt"],
}

TRAIN_FRAC = 0.7    # split on contiguous time blocks, never at random:
                    # adjacent times are strongly autocorrelated and a random
                    # split would leak the test set into training.


def build(tag):
    """Return (X, y, names, aux) for hemisphere tag 'sh' or 'nh'."""
    z = np.load(DATA)
    g = lambda k: z[f"{k}_{tag}"].astype(np.float64)

    tilt_zs, tilt_zl, tilt_zg = g("tilt_zs"), g("tilt_zl"), g("tilt_zg_zs")
    precl, epwp = g("precl"), g("epwp_zl")
    u_zs, v_zs, u_zl, v_zl = g("u_zs"), g("v_zs"), g("u_zl"), g("v_zl")

    cols = {
        "D":   tilt_zs / D_REF,
        "Dl":  tilt_zl / D_REF,
        "Dg":  tilt_zg / D_REF,
        "Dsl": g("tilt_zs_zl") / D_REF,
        "P":   np.maximum(precl / P_REF, P_FLOOR),
        "Us":  np.hypot(u_zs, v_zs) / U_REF,
        "Ul":  np.hypot(u_zl, v_zl) / U_REF,
        "Zs":  g("z_steer") / Z_REF,
        "Zt":  np.abs(g("zeta_zs")) / Zeta_REF,
    }
    names = FEATURE_SETS[MODE]
    X = np.column_stack([cols[n] for n in names])
    y = np.log(epwp / Y_REF)
    return X, y, names, dict(tidx=z[f"tidx_{tag}"], epwp=epwp,
                             tilt_zs=tilt_zs, precl=precl)


def r_log(pred_log, true_log):
    ok = np.isfinite(pred_log) & np.isfinite(true_log)
    return float(np.corrcoef(pred_log[ok], true_log[ok])[0, 1])


def main():
    global MODE, TAG
    MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
    niter = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    assert MODE in FEATURE_SETS, f"mode must be one of {list(FEATURE_SETS)}"
    # Concurrent searches of the same mode (a quick local one alongside the
    # queued PBS job) must not write to the same run directory or front file.
    TAG = MODE + os.environ.get("RUN_TAG", "")

    # Count the cores actually allocated to this process. Neither nproc nor
    # os.cpu_count() is trustworthy here: PBS sets ompthreads=1, which puts
    # OMP_NUM_THREADS=1 in the environment, and GNU nproc honours that and
    # reports 1 even inside a 16-core job. The affinity mask is the truth.
    nproc = int(os.environ.get("PYSR_PROCS", 0)) or len(os.sched_getaffinity(0))
    print(f"mode={MODE}  niterations={niter}  procs={nproc}", flush=True)

    # PYSR_TRAIN=both fits on the two hemispheres together. nn_gwsrc.py shows why
    # this matters: an SH-only fit sends wind-based predictors chasing SH-specific
    # structure -- NH R2 goes negative, and even after an optimal affine
    # recalibration those sets transfer no better than plain (D, P). Trained on
    # both, the same predictors lift NH r from 0.652 to 0.761 and the calibration
    # slope moves from ~0.7 to ~0.92. A global parameterization should be fitted
    # globally.
    train_on = os.environ.get("PYSR_TRAIN", "sh")
    X, y, names, aux = build("sh")
    tidx, hemi = aux["tidx"], np.zeros(X.shape[0], bool)

    if train_on == "both":
        Xn, yn, _, auxn = build("nh")
        X, y = np.concatenate([X, Xn]), np.concatenate([y, yn])
        hemi = np.concatenate([hemi, np.ones(Xn.shape[0], bool)])
        tidx = np.concatenate([tidx, auxn["tidx"]])

    tsplit = np.quantile(tidx, TRAIN_FRAC)
    train = tidx <= tsplit
    print(f"train_on={train_on}: {X.shape[0]:,} samples, "
          f"{len(names)} features {names}")
    print(f"    {train.sum():,} train / {(~train).sum():,} test (time-blocked)")

    # Subsample the training half. Batching already means the search never sees
    # more than batch_size points per comparison; carrying all 3M through Julia
    # buys nothing and costs memory. Compute is better spent on more iterations.
    rng = np.random.default_rng(0)
    tr = np.flatnonzero(train)
    if tr.size > 1_000_000:
        tr = rng.choice(tr, 1_000_000, replace=False)
    Xtr, ytr = X[tr], y[tr]

    os.makedirs(OUTDIR, exist_ok=True)
    model = PySRRegressor(
        niterations=niter,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["exp", "log", "sqrt", "square"],
        populations=3 * nproc,
        population_size=50,
        maxsize=25 if MODE == "dp" else 40,
        parallelism="multithreading",
        procs=nproc,
        turbo=True,
        # Ratio matters more than either number alone. At 2:1 a variable has to
        # earn its place against two constants, which is what keeps the search
        # from assembling baroque products of correlated tilts early on.
        complexity_of_variables=2,
        complexity_of_constants=1,
        # Powers and exponentials are where this search runs away. The nesting
        # bans are what make "^" usable at all; without them the front fills
        # with x^(y^z) towers that fit the batch and nothing else.
        nested_constraints={
            "exp": {"exp": 0, "^": 0, "log": 0},
            "^": {"^": 0, "exp": 0},
            "sqrt": {"sqrt": 0, "log": 0},
            "log": {"exp": 0, "log": 0},
            "square": {"square": 0, "exp": 0},
        },
        constraints={"^": (-1, 1)},
        batching=True,
        batch_size=1000,
        # The real budget is niterations * populations, and at ~120 iterations
        # per minute on 32 cores the settings above would need ~27 h. The tips
        # want niterations large, so bound the wall clock instead of shrinking
        # it: the search stops on the timeout with its hall of fame intact, and
        # the scoring below still runs. Leave headroom under the PBS walltime.
        timeout_in_seconds=int(os.environ.get("PYSR_TIMEOUT", 5 * 3600)),
        output_directory=OUTDIR,
        run_id=f"gwsrc_{TAG}",
        progress=False,
        verbosity=1,
    )

    t0 = time.time()
    model.fit(Xtr, ytr, variable_names=names)
    print(f"search wall time: {(time.time() - t0) / 60:.1f} min", flush=True)

    # Score every member of the Pareto front, not just the selected one: the
    # point of the front is to read the accuracy/complexity tradeoff, and the
    # default pick is a heuristic, not an answer.
    #
    # r_sh and r_nh are always held-out *in time*. Under train_on='sh' the NH is
    # additionally unseen in space, so it is a transfer test; under 'both' it is
    # only a time test. Two different questions, so the column is labelled.
    if train_on == "both":
        Xsh, ysh = X[~train & ~hemi], y[~train & ~hemi]
        Xnh, ynh = X[~train & hemi], y[~train & hemi]
    else:
        Xsh, ysh = X[~train], y[~train]
        Xnh, ynh, _, _ = build("nh")

    rows = []
    for i in range(len(model.equations_)):
        eq = model.equations_.iloc[i]
        try:
            p_tr = model.predict(X[train], index=i)
            p_sh = model.predict(Xsh, index=i)
            p_nh = model.predict(Xnh, index=i)
        except Exception as e:                       # noqa: BLE001
            print(f"  eq {i} failed to evaluate: {e}")
            continue
        rows.append(dict(
            index=i, complexity=int(eq["complexity"]), loss=float(eq["loss"]),
            equation=str(eq["equation"]),
            r_train=r_log(p_tr, y[train]),
            r_test=r_log(p_sh, ysh),
            r_nh=r_log(p_nh, ynh),
        ))

    print(f"\n{'cx':>3} {'r_train':>8} {'r_sh':>8} {'r_nh':>8}  equation")
    for r in rows:
        print(f"{r['complexity']:3d} {r['r_train']:8.4f} {r['r_test']:8.4f} "
              f"{r['r_nh']:8.4f}  {r['equation'][:110]}")

    with open(os.path.join(OUTDIR, f"front_{TAG}.json"), "w") as f:
        json.dump(dict(mode=TAG, names=names, rows=rows, train_on=train_on,
                       sigma=SIGMA,
                       refs=dict(D_REF=D_REF, P_REF=P_REF, Y_REF=Y_REF,
                                 U_REF=U_REF, Z_REF=Z_REF, Zeta_REF=Zeta_REF,
                                 P_FLOOR=P_FLOOR)), f, indent=1)
    print(f"\nwrote {OUTDIR}/front_{TAG}.json")


if __name__ == "__main__":
    main()
