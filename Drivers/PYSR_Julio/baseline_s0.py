"""
Re-baseline the project on bare f09 columns (dlon 1.25, dlat 0.9424, sigma=0).

That grid is the parameterization's grid, so it is where the incumbent has to be
scored and where the ceiling has to be computed. Everything published so far --
Julio's 0.5784, our 0.70 fronts -- was measured on sigma=2 smoothed fields,
which discard ~38% of the log-variance a column actually has (smoothing_ladder.py).

Reports four things at sigma=0, each against sigma=2 for reference:
  1. Julio's 7-parameter form with its published coefficients, i.e. what CAM
     would get if the current fit were dropped in as-is.
  2. The same form refit at sigma=0 -- the form's honest per-column best.
  3. The variance-decomposition ceiling on f(D, P), the bound that any function
     of those two inputs obeys.
  4. The same ceiling for the six-predictor set, as a form-free reference.

    conda activate npl-2026a && PYTHONNOUSERSITE=1 python baseline_s0.py
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/glade/work/juliob/HiRes_ana_dev/Drivers/Analysis")
import distro_fitting as DF        # noqa: E402
import source_fit_refvals as SF    # noqa: E402

BASE = "/glade/work/wchapman/PYSR_Julio"
OUT = f"{BASE}/baseline_s0.json"

# BasicAna4GWP-fitting.ipynb cell 28: the NNLS grid solution, fit on sigma=2.
PAR_JULIO = dict(y0=0.0, A=0.05662744099311622, a=1.5,
                 D0=4.754679577383344e-08, B=0.002615114957098987,
                 b=1.0999999999999999, c=0.30000000000000004)

SCALES = [("sigma=2 (as published)", "gwsrc_data.npz"),
          ("sigma=0 (bare f09 column)", "gwsrc_data_s0.npz")]


def load(path, tag):
    z = np.load(f"{BASE}/{path}")
    D = z[f"tilt_zs_{tag}"].astype(np.float64)
    P = z[f"precl_{tag}"].astype(np.float64)
    Y = z[f"epwp_zl_{tag}"].astype(np.float64)
    ok = (D > 0) & (P > 0) & (Y > 0)
    return D[ok], P[ok], Y[ok], z, ok


def r_log(pred, true):
    return float(np.corrcoef(np.log(pred), np.log(true))[0, 1])


def ceiling(D, P, Y, nd=40, npp=40):
    """Variance-decomposition bound on any f(D, P), via Julio's own
    `distro_fitting.cell_spread` so the number is computed the same way it was
    for the sigma=2 result (0.583). Quantile edges keep cells populated."""
    d_edges = np.quantile(D, np.linspace(0, 1, nd + 1))
    p_edges = np.quantile(P, np.linspace(0, 1, npp + 1))
    res = DF.cell_spread(D, P, Y, d_edges, p_edges)
    return float(res["r_ceiling"]), int(np.isfinite(res["cell_mean"]).sum())


def main():
    out = {}
    for label, path in SCALES:
        print(f"\n=== {label} ===")
        rec = {}
        for tag in ("sh", "nh"):
            D, P, Y, _, _ = load(path, tag)
            pred = SF.source_model(D, P, **PAR_JULIO)
            r_pub = r_log(pred, Y)

            # Refit the same form at this scale, in log space.
            par_fit, r_fit = SF.fit_log(D, P, Y, PAR_JULIO) \
                if hasattr(SF, "fit_log") else (None, None)
            if par_fit is None:
                par_fit, r_fit = refit(D, P, Y)

            r_ceil, ncell = ceiling(D, P, Y)
            rec[tag] = dict(n=int(Y.size), r_published=r_pub, r_refit=r_fit,
                            r_ceiling_dp=r_ceil, n_cells=ncell,
                            var_log_target=float(np.log(Y).var()),
                            par_refit=par_fit)
            print(f"  [{tag}] n={Y.size:>9,}  var(log tau)={np.log(Y).var():.3f}")
            print(f"        Julio's form, published coeffs : r={r_pub:.4f}")
            print(f"        Julio's form, refit here       : r={r_fit:.4f}")
            print(f"        ceiling for ANY f(D,P)         : r={r_ceil:.4f}"
                  f"  ({ncell} cells)")
        out[label] = rec

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


def refit(D, P, Y, n=400_000, seed=0):
    """Least-squares refit of the 7-parameter form on log(Y), on a subsample."""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    i = rng.choice(Y.size, min(n, Y.size), replace=False)
    d, p, ly = D[i], P[i], np.log(Y[i])
    keys = ["A", "a", "D0", "B", "b", "c"]
    x0 = np.array([np.log(PAR_JULIO["A"]), PAR_JULIO["a"],
                   np.log(PAR_JULIO["D0"]), np.log(PAR_JULIO["B"]),
                   PAR_JULIO["b"], PAR_JULIO["c"]])

    def unpack(x):
        return dict(y0=0.0, A=np.exp(x[0]), a=x[1], D0=np.exp(x[2]),
                    B=np.exp(x[3]), b=x[4], c=x[5])

    def resid(x):
        pred = SF.source_model(d, p, **unpack(x))
        return np.log(np.maximum(pred, 1e-30)) - ly

    sol = least_squares(resid, x0, method="trf", max_nfev=200)
    par = unpack(sol.x)
    return {k: float(par[k]) for k in ["y0"] + keys}, \
        r_log(SF.source_model(D, P, **par), Y)


if __name__ == "__main__":
    main()
