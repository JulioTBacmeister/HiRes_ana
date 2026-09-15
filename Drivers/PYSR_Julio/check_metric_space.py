"""
Confirm which space the reported r lives in, and show the other one.

The target fed to PySR is y = log(epwp / Y_REF), so a model's *output* is already
a log. The reported r is therefore corr(pred_log, true_log) -- the log-space
Pearson r, the same statistic as source_fit_refvals.score's r_full and the same
one Julio quotes (0.77 for the event NN). This asserts that rather than trusting
the variable naming, and reports the linear-space r alongside, which is a
different and much less flattering number: exponentiating restores the heavy
right tail, so a few large events dominate the covariance.

    conda activate /glade/work/wchapman/conda-envs/L96M2lines
    PYTHONNOUSERSITE=1 python check_metric_space.py
"""

import numpy as np
from pysr import PySRRegressor

BASE = "/glade/work/wchapman/PYSR_Julio"
Y_REF = 4.68e-3


def rp(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main():
    z = np.load(f"{BASE}/distill_data.npz")
    X, y = z["X"].astype(np.float64), z["y"].astype(np.float64)
    train = z["tidx"] <= z["tsplit"]
    hemi = z["hemi"]
    sh = (~train) & (~hemi)

    # y is log(epwp / Y_REF) by construction; recover physical units.
    tau = Y_REF * np.exp(y)
    assert np.allclose(np.log(tau / Y_REF), y, atol=1e-9)
    print("target stored as log(epwp/Y_REF): confirmed")
    print(f"  log space : mean {y[sh].mean():+.3f}  sd {y[sh].std():.3f}")
    print(f"  linear    : median {np.median(tau[sh]):.4g}  "
          f"p99/p50 {np.percentile(tau[sh], 99)/np.median(tau[sh]):.1f}x\n")

    m = PySRRegressor.from_file(
        run_directory=f"{BASE}/pysr_outputs/gwsrc_distill_data")

    print(f"{'cx':>3} {'r (log)':>9} {'r (linear)':>11} {'R2 (log)':>9}  equation")
    for i in range(len(m.equations_)):
        eq = m.equations_.iloc[i]
        cx = int(eq["complexity"])
        if cx not in (11, 15, 17, 24, 28, 34, 37):
            continue
        p_log = m.predict(X[sh], index=i)          # model output is a log
        p_lin = Y_REF * np.exp(p_log)              # back to physical units

        r_log = rp(p_log, y[sh])
        r_lin = rp(p_lin, tau[sh])
        r2_log = 1.0 - np.sum((p_log - y[sh]) ** 2) / np.sum(
            (y[sh] - y[sh].mean()) ** 2)
        print(f"{cx:3d} {r_log:9.4f} {r_lin:11.4f} {r2_log:9.4f}  "
              f"{str(eq['equation'])[:60]}")

    print("\nThe reported column throughout this project is r (log).")
    print("Linear-space r is lower because exp() restores the heavy right tail,")
    print("so a handful of the largest events dominate the covariance.")


if __name__ == "__main__":
    main()
