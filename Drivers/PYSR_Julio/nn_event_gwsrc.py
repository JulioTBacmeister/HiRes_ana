"""
Stage 2: fit Julio's MLP on the event-composite predictors and score it the way
we score everything else in this directory.

Runs Julio's own `mlp_utils.fit_mlp_general` -- not a reimplementation -- so the
numbers are his architecture, his split, his loss. What is added is the scoring:
his function reports r and R2 in *linear* space, and the notebook computes only
a log-space r for the transfer. Here every model gets the full log-space panel
(r, R2, calibration slope, median bias in dex) on SH-train, SH-test and NH, so
the result sits directly alongside the PySR fronts and the gridded MLPs.

Two configurations are compared:
  - loss_power=5, Julio's default. Weights the largest events ~5th power.
  - loss_power=2, plain MSE on standardised log(target).
CLAUDE.md flags the |error|^5 loss as the likely cause of the r^2-vs-R2 gap in
his sweep; this measures whether that is true.

    conda activate /glade/work/wchapman/conda-envs/credit-casper-modern
    PYTHONNOUSERSITE=1 python nn_event_gwsrc.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/glade/work/juliob/HiRes_ana_dev/Drivers/Analysis")
import mlp_utils as mlu   # noqa: E402  (standalone: torch/sklearn/scipy only)

DATA = "/glade/work/wchapman/PYSR_Julio/event_data.npz"
OUT = "/glade/work/wchapman/PYSR_Julio/event_nn_results.json"

SETS = ["tilt_levs+tilt", "tilt_levs+precl", "tilt_levs+tilt+precl",
        "tilt_levs+fgf", "tilt_levs+tilt+fgf"]
BEST = "tilt_levs+tilt+precl"

# sweep3 hyperparameters, merged over predictors.reset_hyperparameters().
HP = dict(
    hidden_dims=(64, 64),
    dropout=0.1,
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-5,
    patience=100,
    max_epochs=500,
    lr_patience=15,
    lr_factor=0.5,
    min_lr=1e-6,
    loss_power=5,
    log_predictor_patterns=["tilt", "U_wv_src_mm", "abs_zeta"],
)
TRAIN_INTERVAL = (48, 248)
TEST_INTERVAL = (0, 28)


def metrics(pred, true):
    """Log-space skill panel. `pred` and `true` are in physical units.

    R2_cal is R2 after the best affine correction in log space and must equal
    r^2 exactly -- asserted, because a float32 least-squares fit on this many
    points silently returns a wrong slope and produced a wrong conclusion here
    once already.
    """
    eps = 1e-30
    p = np.log(np.maximum(np.asarray(pred, np.float64), eps))
    t = np.log(np.maximum(np.asarray(true, np.float64), eps))

    r = float(np.corrcoef(p, t)[0, 1])
    denom = float(np.sum((t - t.mean()) ** 2))
    r2 = float(1.0 - np.sum((p - t) ** 2) / denom)

    pm, tm = p.mean(), t.mean()
    slope = float(np.sum((p - pm) * (t - tm)) / np.sum((p - pm) ** 2))
    offset = float(tm - slope * pm)
    r2_cal = float(1.0 - np.sum((slope * p + offset - t) ** 2) / denom)
    assert abs(r2_cal - r * r) < 1e-8, f"R2_cal {r2_cal} != r^2 {r * r}"

    return dict(n=int(p.size), r=r, r2=r2, r2_cal=r2_cal, slope=slope,
                bias_dex=float(np.median(p - t) / np.log(10.0)))


def run(d, set_name, loss_power, train_on="sh"):
    """Fit one model and score it on SH-train, SH-test and the NH hemisphere."""
    X = d[f"X_sh_{set_name}"]
    y = d["y_sh"].astype(np.float64)
    t = d["tidx_sh"]

    if train_on == "both":
        # Concatenate hemispheres before the split, so the time-block split
        # applies to both. The NH column then stops being a transfer test and
        # becomes a held-out-time test -- record which one it was.
        X = np.concatenate([X, d[f"X_nh_{set_name}"]])
        y = np.concatenate([y, d["y_nh"].astype(np.float64)])
        t = np.concatenate([t, d["tidx_nh"]])

    keep = y > 0
    if not keep.all():
        print(f"  dropping {(~keep).sum()} non-positive targets")
        X, y, t = X[keep], y[keep], t[keep]

    hp = dict(HP, loss_power=loss_power)
    model, res = mlu.fit_mlp_general(
        predictors=[X[:, i] for i in range(X.shape[1])],
        predictor_names=list(d[f"names_sh_{set_name}"]),
        target=y,
        event_times=t,
        train_interval=TRAIN_INTERVAL,
        test_interval=TEST_INTERVAL,
        verbose=False,
        **hp,
    )

    meta = {k: res[k] for k in ("feat_scaler", "target_scaler",
                                "log_predictor_patterns", "log_pred_mask")}

    # apply_mlp moves its input to `device`, so it must be the device the model
    # is actually on -- fit_mlp_general picks cuda when it is available, while
    # the notebook hard-codes 'cpu' and only works because it runs CPU-only.
    import torch
    device = str(next(model.parameters()).device)

    Xn = d[f"X_nh_{set_name}"]
    yn = d["y_nh"].astype(np.float64)
    keepn = yn > 0
    pred_nh = mlu.apply_mlp(model, meta, device,
                            [Xn[keepn][:, i] for i in range(Xn.shape[1])])

    out = {
        "set": set_name,
        "loss_power": loss_power,
        "train_on": train_on,
        "n_predictors": int(X.shape[1]),
        "train": metrics(res["y_pred_train"], res["y_train"]),
        "test": metrics(res["y_pred_test"], res["y_test"]),
        "nh": metrics(pred_nh, yn[keepn]),
    }
    # Julio's own linear-space numbers, for cross-checking against his sweep.
    out["linear_test_r"] = float(res["r_test"])
    out["linear_test_r2"] = float(res["r2_test"])

    imp = np.asarray(res["importances"], float)
    order = np.argsort(imp)[::-1]
    names = list(d[f"names_sh_{set_name}"])
    out["importance"] = [(names[i], float(imp[i])) for i in order]
    return out


def fmt(tag, m):
    return (f"{tag:>10s}  n={m['n']:>6d}  r={m['r']:.3f}  R2={m['r2']:+.3f}  "
            f"slope={m['slope']:.2f}  bias={m['bias_dex']:+.2f} dex")


def main():
    d = np.load(DATA, allow_pickle=True)
    print(f"SH events: {d['y_sh'].size}   NH events: {d['y_nh'].size}")

    results = []
    print("\n=== Julio's configuration: SH-fit, |error|^5 loss ===")
    for s in SETS:
        r = run(d, s, loss_power=5, train_on="sh")
        results.append(r)
        print(f"\n{s}  ({r['n_predictors']} predictors)")
        for k in ("train", "test", "nh"):
            print("  " + fmt(k, r[k]))
        print(f"  (his linear-space test r={r['linear_test_r']:.3f} "
              f"R2={r['linear_test_r2']:+.3f})")

    print("\n=== calibration test: best set under plain MSE ===")
    r = run(d, BEST, loss_power=2, train_on="sh")
    results.append(r)
    for k in ("train", "test", "nh"):
        print("  " + fmt(k, r[k]))

    print("\n=== global fit: best set, both hemispheres ===")
    for lp in (5, 2):
        r = run(d, BEST, loss_power=lp, train_on="both")
        results.append(r)
        print(f"  loss_power={lp}")
        for k in ("train", "test", "nh"):
            print("    " + fmt(k, r[k]))

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
