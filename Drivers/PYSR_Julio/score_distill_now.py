"""
Score the in-flight distillation fronts against real data, without waiting.

`run_pysr_distill.py` only scores at the end of its 5 h budget. PySR writes its
hall of fame continuously, so this reads the checkpoints mid-run and applies the
same rule: **every equation is scored against the real target**, on held-out
times in each hemisphere, never against the teacher it was fit to.

    conda activate /glade/work/wchapman/conda-envs/L96M2lines
    PYTHONNOUSERSITE=1 python score_distill_now.py
"""

import os

import numpy as np
from pysr import PySRRegressor

BASE = "/glade/work/wchapman/PYSR_Julio"
OUTDIR = f"{BASE}/pysr_outputs"
TEACHERS = ["mlp", "tree", "data"]


def r_log(p, t):
    ok = np.isfinite(p) & np.isfinite(t)
    return float(np.corrcoef(p[ok], t[ok])[0, 1]) if ok.sum() > 2 else np.nan


def main():
    z = np.load(f"{BASE}/distill_data.npz")
    X, y = z["X"].astype(np.float64), z["y"].astype(np.float64)
    tidx, hemi = z["tidx"], z["hemi"]
    train = tidx <= z["tsplit"]
    sh_te, nh_te = (~train) & (~hemi), (~train) & hemi
    print(f"test rows: SH {sh_te.sum():,}  NH {nh_te.sum():,}\n")

    for t in TEACHERS:
        run = f"{OUTDIR}/gwsrc_distill_{t}"
        if not os.path.exists(f"{run}/hall_of_fame.csv"):
            print(f"[{t}] no hall of fame yet\n")
            continue
        try:
            m = PySRRegressor.from_file(run_directory=run)
        except Exception as e:                          # noqa: BLE001
            print(f"[{t}] could not load: {e}\n")
            continue

        target = {"data": y, "tree": z["y_tree"].astype(np.float64),
                  "mlp": z["y_mlp"].astype(np.float64)}[t]

        print(f"===== teacher = {t} =====")
        print(f"{'cx':>3} {'SH':>7} {'NH':>7} {'r_teach':>8}  equation")
        best = None
        for i in range(len(m.equations_)):
            eq = m.equations_.iloc[i]
            try:
                p_sh = m.predict(X[sh_te], index=i)
                p_nh = m.predict(X[nh_te], index=i)
            except Exception:                           # noqa: BLE001
                continue
            r_sh, r_nh = r_log(p_sh, y[sh_te]), r_log(p_nh, y[nh_te])
            r_tc = r_log(p_sh, target[sh_te])
            cx = int(eq["complexity"])
            if cx >= 10:
                print(f"{cx:3d} {r_sh:7.4f} {r_nh:7.4f} {r_tc:8.4f}  "
                      f"{str(eq['equation'])[:88]}")
            if best is None or r_sh > best[1]:
                best = (cx, r_sh, r_nh, str(eq["equation"]))
        if best:
            print(f"\n  best-SH: cx={best[0]}  SH={best[1]:.4f}  NH={best[2]:.4f}")
        print()

    print("reference   direct fit (full_both, 5 h): SH 0.6949  NH 0.6979")
    print("reference   MLP teacher                : SH 0.7333  NH 0.7337")
    print("reference   tree teacher               : SH 0.7297  NH 0.7340")


if __name__ == "__main__":
    main()
