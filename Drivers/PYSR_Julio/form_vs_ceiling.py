"""
Is the *equation* the limiting factor, or the *features*?

This is the question that decides whether distilling Julio's NN is worth doing.
Distillation can only ever transfer what a flexible model knows that a symbolic
one does not. So: give a form-free model exactly the feature set, scaling,
split and target that `run_pysr_gwsrc.py` gave PySR, and compare.

  tree >> PySR  -> the symbolic search is leaving skill on the table, and
                   distilling a flexible model (his NN, or this tree) as a
                   denoised target is a real option.
  tree ~= PySR  -> PySR has already extracted everything these features carry.
                   Distillation cannot add skill, only inherit the teacher's
                   errors, and the payoff must come from better features.

Uses the same 2:1 pattern as the rest of the directory: HistGradientBoosting
stands in for "any model of these inputs".

    conda activate npl-2026a && PYTHONNOUSERSITE=1 python form_vs_ceiling.py
"""

import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

BASE = "/glade/work/wchapman/PYSR_Julio"
SIGMA = float(os.environ.get("GWSRC_SIGMA", 2))
DATA = f"{BASE}/gwsrc_data_s{SIGMA:g}.npz"
OUT = f"{BASE}/form_vs_ceiling.json"

# Copied from run_pysr_gwsrc.py -- must stay identical or the comparison is void.
D_REF, P_REF, Y_REF = 3e-7, 3e-8, 4.68e-3
U_REF, Z_REF, Zeta_REF = 10.0, 5000.0, 3e-5
P_FLOOR = 1e-4
TRAIN_FRAC = 0.7
FEATURE_SETS = {
    "dp":   ["D", "P"],
    "tilt": ["D", "Dl", "Dg", "Dsl", "P"],
    "full": ["D", "Dl", "Dg", "P", "Us", "Ul", "Zs", "Zt"],
}
# What the PySR fronts reached, for the same mode/train_on. Read from the JSONs.
FRONTS = {("dp", "both"): "front_dp_both.json",
          ("tilt", "both"): "front_tilt_both.json",
          ("full", "both"): "front_full_both.json",
          ("dp", "sh"): "front_dp.json",
          ("tilt", "sh"): "front_tilt.json"}


def build(z, tag, names):
    g = lambda k: z[f"{k}_{tag}"].astype(np.float64)
    cols = {
        "D":   g("tilt_zs") / D_REF,
        "Dl":  g("tilt_zl") / D_REF,
        "Dg":  g("tilt_zg_zs") / D_REF,
        "Dsl": g("tilt_zs_zl") / D_REF,
        "P":   np.maximum(g("precl") / P_REF, P_FLOOR),
        "Us":  np.hypot(g("u_zs"), g("v_zs")) / U_REF,
        "Ul":  np.hypot(g("u_zl"), g("v_zl")) / U_REF,
        "Zs":  g("z_steer") / Z_REF,
        "Zt":  np.abs(g("zeta_zs")) / Zeta_REF,
    }
    X = np.column_stack([cols[n] for n in names])
    y = np.log(g("epwp_zl") / Y_REF)
    return X, y, z[f"tidx_{tag}"]


def r_log(p, t):
    ok = np.isfinite(p) & np.isfinite(t)
    return float(np.corrcoef(p[ok], t[ok])[0, 1])


def pysr_best(mode, train_on):
    """Best r_test on the front, and its NH score."""
    path = f"{BASE}/pysr_outputs/{FRONTS[(mode, train_on)]}"
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        return None
    row = max(d["rows"], key=lambda r: r.get("r_test", -9))
    return dict(complexity=row["complexity"], r_test=row["r_test"],
                r_nh=row["r_nh"])


def main():
    z = np.load(DATA)
    rng = np.random.default_rng(0)
    results = []

    for mode in ("dp", "tilt", "full"):
        for train_on in ("sh", "both"):
            if (mode, train_on) not in FRONTS:
                continue
            names = FEATURE_SETS[mode]
            Xs, ys, ts = build(z, "sh", names)
            X, y, tidx = Xs, ys, ts
            if train_on == "both":
                Xn, yn, tn = build(z, "nh", names)
                X = np.concatenate([X, Xn])
                y = np.concatenate([y, yn])
                tidx = np.concatenate([tidx, tn])

            tsplit = np.quantile(tidx, TRAIN_FRAC)
            train = tidx <= tsplit
            tr = np.flatnonzero(train)
            if tr.size > 1_000_000:            # same cap PySR gets
                tr = rng.choice(tr, 1_000_000, replace=False)

            tree = HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.08, max_leaf_nodes=127,
                early_stopping=True, random_state=0)
            tree.fit(X[tr], y[tr])

            te = np.flatnonzero(~train)
            Xn2, yn2, tn2 = build(z, "nh", names)
            nh = tn2 > np.quantile(ts, TRAIN_FRAC)   # NH, held-out times only

            rec = dict(mode=mode, train_on=train_on, n_features=len(names),
                       tree_r_test=r_log(tree.predict(X[te]), y[te]),
                       tree_r_nh=r_log(tree.predict(Xn2[nh]), yn2[nh]),
                       pysr=pysr_best(mode, train_on))
            results.append(rec)
            p = rec["pysr"]
            print(f"{mode:>5s} {train_on:>5s} {len(names)}f   "
                  f"tree SH={rec['tree_r_test']:.3f} NH={rec['tree_r_nh']:.3f}   "
                  f"PySR SH={p['r_test']:.3f} NH={p['r_nh']:.3f} (cx={p['complexity']})   "
                  f"gap SH={p['r_test'] - rec['tree_r_test']:+.3f} "
                  f"NH={p['r_nh'] - rec['tree_r_nh']:+.3f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}")
    print("\ngap > 0 means the equation BEATS the form-free model, so there is "
          "nothing for a teacher to distil.")


if __name__ == "__main__":
    main()
