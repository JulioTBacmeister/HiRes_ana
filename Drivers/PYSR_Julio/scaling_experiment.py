"""
Which input scaling should the GW-source search use?

pysrtips_bety260326.py is explicit that this is not a cosmetic choice:

    "experiment with different variable scalings. E.g. z-scoring a strictly
     positive variable x1 might get rid of any x1^a where a<1. In theory the
     algorithm should recover (x1+offset)^a but in practice it doesn't always
     and so scaling choices can change the form and performance of the
     equations"

So test it rather than assert it. Three scalings of the same eight predictors,
same seed, same budget, same everything else:

  ratio   x / x_ref, x_ref near the median. Strictly positive, power laws
          survive intact. What run_pysr_gwsrc.py uses.
  zscore  (x - mean) / std. The tips' warning case: variables go negative, so
          x^a with fractional a leaves the reals and the search has to rebuild
          every power law as something else.
  log     log(x / x_ref). The target is already log tau, so a power law becomes
          a *linear* term -- the best-conditioned option in principle, and the
          one that gives up multiplicative structure in exchange.

Short runs on purpose: this compares conditioning, not converged skill. The tips
warn that small runs do not extrapolate, so the winner here sets the scaling for
a long run, it does not settle the equation.

    conda activate /glade/work/wchapman/conda-envs/L96M2lines
    python scaling_experiment.py [niterations]
"""

import os
import sys
import json
import time

import numpy as np
from pysr import PySRRegressor

import run_pysr_gwsrc as R

OUT = os.path.join(R.OUTDIR, "scaling_experiment.json")


def features(scaling, rng):
    """Eight predictors under one scaling, plus the log-tau target."""
    z = np.load(R.DATA)
    g = lambda k: z[f"{k}_sh"].astype(np.float64)
    u_zs, v_zs, u_zl, v_zl = g("u_zs"), g("v_zs"), g("u_zl"), g("v_zl")

    # Natural units first; the scaling is applied to these.
    raw = {
        "D":  g("tilt_zs") / R.D_REF,
        "Dl": g("tilt_zl") / R.D_REF,
        "Dg": g("tilt_zg_zs") / R.D_REF,
        "P":  np.maximum(g("precl") / R.P_REF, R.P_FLOOR),
        "Us": np.hypot(u_zs, v_zs) / R.U_REF,
        "Ul": np.hypot(u_zl, v_zl) / R.U_REF,
        "Zs": g("z_steer") / R.Z_REF,
        "Zt": np.abs(g("zeta_zs")) / R.Zeta_REF,
    }
    names = list(raw)
    X = np.column_stack([raw[n] for n in names])

    if scaling == "ratio":
        pass
    elif scaling == "zscore":
        X = (X - X.mean(0)) / X.std(0)
    elif scaling == "log":
        X = np.log(X)
    else:
        raise ValueError(scaling)

    y = np.log(g("epwp_zl") / R.Y_REF)
    return X, y, names, z["tidx_sh"]


def main():
    niter = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    nproc = int(os.environ.get("PYSR_PROCS", 0)) or len(os.sched_getaffinity(0))
    rng = np.random.default_rng(0)

    results = {}
    for scaling in ("ratio", "zscore", "log"):
        X, y, names, tidx = features(scaling, rng)
        train = tidx <= np.quantile(tidx, R.TRAIN_FRAC)

        # Identical subsample across scalings so the comparison is of the
        # scaling and nothing else.
        tr = np.flatnonzero(train)
        tr = np.random.default_rng(0).choice(tr, 300_000, replace=False)

        print(f"\n{'=' * 70}\n{scaling}\n{'=' * 70}", flush=True)
        print(f"{'feature':8s} {'1%':>10s} {'median':>10s} {'99%':>10s}")
        for j, n in enumerate(names):
            q = np.percentile(X[:, j], [1, 50, 99])
            print(f"{n:8s} {q[0]:10.3g} {q[1]:10.3g} {q[2]:10.3g}")

        model = PySRRegressor(
            niterations=niter,
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["exp", "log", "sqrt", "square"],
            populations=3 * nproc,
            population_size=50,
            maxsize=30,
            parallelism="multithreading",
            procs=nproc,
            turbo=True,
            complexity_of_variables=2,
            complexity_of_constants=1,
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
            output_directory=R.OUTDIR,
            run_id=f"scaling_{scaling}",
            deterministic=False,
            random_state=0,
            progress=False,
            verbosity=0,
        )

        t0 = time.time()
        model.fit(X[tr], y[tr], variable_names=names)
        wall = (time.time() - t0) / 60

        rows = []
        for i in range(len(model.equations_)):
            eq = model.equations_.iloc[i]
            try:
                p_te = model.predict(X[~train], index=i)
            except Exception as e:                        # noqa: BLE001
                print(f"  eq {i} failed: {e}")
                continue
            ok = np.isfinite(p_te)
            if ok.sum() < 100:
                continue
            rows.append(dict(
                complexity=int(eq["complexity"]),
                r_test=float(np.corrcoef(p_te[ok], y[~train][ok])[0, 1]),
                n_bad=int((~ok).sum()),
                equation=str(eq["equation"]),
            ))

        best = max(rows, key=lambda r: r["r_test"]) if rows else None
        results[scaling] = dict(wall_min=wall, rows=rows, best=best)
        print(f"\nwall {wall:.1f} min   front {len(rows)}")
        if best:
            print(f"best r_test = {best['r_test']:.4f} at complexity "
                  f"{best['complexity']}\n  {best['equation'][:140]}")

    print(f"\n{'=' * 70}\n{'scaling':10s} {'best r_test':>12s} {'cx':>4s} "
          f"{'non-finite':>11s}")
    for k, v in results.items():
        b = v["best"]
        if b:
            print(f"{k:10s} {b['r_test']:12.4f} {b['complexity']:4d} "
                  f"{b['n_bad']:11d}")

    with open(OUT, "w") as f:
        json.dump(dict(niterations=niter, results=results), f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    R.MODE = "full"
    main()
