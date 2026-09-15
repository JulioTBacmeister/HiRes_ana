"""
Did distillation actually help? A front-level comparison, not a one-number one.

The first read of these runs compared only each front's best-SH equation and
concluded "control matches treatment". That is one order statistic out of ~30
per front, chosen by the noisiest possible rule, and it hid whatever is going on
in the NH. This does it properly:

  - every equation scored against REAL data, SH-test and NH-test
  - paired at matched complexity, so the comparison is like-for-like
  - the right baseline is the `data` control, which is the same script, same
    budget, same split, fitting the noisy target. NOT the older full_both run,
    whose 0.6949 differs from the control by ordinary run-to-run variance in a
    stochastic search.

    conda activate /glade/work/wchapman/conda-envs/L96M2lines
    PYTHONNOUSERSITE=1 python compare_distill.py
"""

import json

import numpy as np
from pysr import PySRRegressor

BASE = "/glade/work/wchapman/PYSR_Julio"
RUNS = ["data", "tree", "mlp"]


def rp(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan


def main():
    z = np.load(f"{BASE}/distill_data.npz")
    X, y = z["X"].astype(np.float64), z["y"].astype(np.float64)
    train = z["tidx"] <= z["tsplit"]
    hemi = z["hemi"]
    sh, nh = (~train) & (~hemi), (~train) & hemi

    fronts = {}
    for t in RUNS:
        m = PySRRegressor.from_file(
            run_directory=f"{BASE}/pysr_outputs/gwsrc_distill_{t}")
        rows = {}
        for i in range(len(m.equations_)):
            eq = m.equations_.iloc[i]
            try:
                rows[int(eq["complexity"])] = dict(
                    sh=rp(m.predict(X[sh], index=i), y[sh]),
                    nh=rp(m.predict(X[nh], index=i), y[nh]),
                    eq=str(eq["equation"]))
            except Exception:                            # noqa: BLE001
                continue
        fronts[t] = rows
        print(f"{t}: {len(rows)} equations")

    # ---- front-level summaries -------------------------------------------
    print(f"\n{'run':>6}  {'best SH':>8} {'best NH':>8}  "
          f"{'mean SH cx>=25':>14} {'mean NH cx>=25':>14}")
    for t in RUNS:
        r = fronts[t]
        big = [v for c, v in r.items() if c >= 25]
        print(f"{t:>6}  {max(v['sh'] for v in r.values()):8.4f} "
              f"{max(v['nh'] for v in r.values()):8.4f}  "
              f"{np.mean([v['sh'] for v in big]):14.4f} "
              f"{np.mean([v['nh'] for v in big]):14.4f}")

    # ---- paired, matched complexity --------------------------------------
    common = sorted(set(fronts["data"]) & set(fronts["tree"]) & set(fronts["mlp"]))
    print(f"\nmatched complexities: {len(common)}")
    print(f"{'cx':>3} | {'data SH':>8} {'tree SH':>8} {'mlp SH':>8} "
          f"| {'data NH':>8} {'tree NH':>8} {'mlp NH':>8}")
    for c in common:
        d, tr, ml = fronts["data"][c], fronts["tree"][c], fronts["mlp"][c]
        print(f"{c:3d} | {d['sh']:8.4f} {tr['sh']:8.4f} {ml['sh']:8.4f} "
              f"| {d['nh']:8.4f} {tr['nh']:8.4f} {ml['nh']:8.4f}")

    print(f"\n{'paired mean difference vs the data control (n=%d)' % len(common)}")
    out = {}
    for t in ("tree", "mlp"):
        dsh = np.array([fronts[t][c]["sh"] - fronts["data"][c]["sh"] for c in common])
        dnh = np.array([fronts[t][c]["nh"] - fronts["data"][c]["nh"] for c in common])
        # SE of the paired mean; the equations at different cx are far from
        # independent, so treat this as optimistic.
        print(f"  {t:>4}:  dSH = {dsh.mean():+.4f} +- {dsh.std(ddof=1)/np.sqrt(len(dsh)):.4f}"
              f"   dNH = {dnh.mean():+.4f} +- {dnh.std(ddof=1)/np.sqrt(len(dnh)):.4f}")
        out[t] = dict(dSH=float(dsh.mean()), dNH=float(dnh.mean()),
                      dSH_se=float(dsh.std(ddof=1)/np.sqrt(len(dsh))),
                      dNH_se=float(dnh.std(ddof=1)/np.sqrt(len(dnh))))

    # How often is each run's equation literally identical to the control's?
    same = {t: sum(fronts[t][c]["eq"] == fronts["data"][c]["eq"] for c in common)
            for t in ("tree", "mlp")}
    print(f"\nidentical equation to the control at matched cx: "
          f"tree {same['tree']}/{len(common)}, mlp {same['mlp']}/{len(common)}")

    with open(f"{BASE}/compare_distill.json", "w") as f:
        json.dump(dict(paired=out, identical=same,
                       fronts={t: {str(c): v for c, v in fronts[t].items()}
                               for t in RUNS}), f, indent=1)
    print(f"\nwrote {BASE}/compare_distill.json")


if __name__ == "__main__":
    main()
