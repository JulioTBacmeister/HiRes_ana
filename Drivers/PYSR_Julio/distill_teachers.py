"""
Train the two distillation teachers on the exact PySR problem, and save what
they predict.

`form_vs_ceiling.py` showed a boosted tree reaches 0.730/0.735 on the `full`
8-feature set where the best PySR equation reaches 0.695/0.698. That 0.036 is
the only measured evidence the symbolic search is leaving anything behind, so
this asks whether it can be recovered as a compact expression.

Two teachers, because they fail differently:

  tree  HistGradientBoosting, identical to form_vs_ceiling.py. Accurate, but its
        output is *piecewise constant* -- ~500 x 127 leaves means tens of
        thousands of discontinuities. A smooth expression cannot track a
        staircase, so complexity spent chasing it is wasted.
  mlp   The nn_gwsrc.py architecture, smooth and differentiable. If it matches
        the tree's skill it is the better teacher by construction; if it lands
        well short, the tree's advantage was probably staircase memorisation and
        the whole distillation idea dies here, cheaply.

Writes `distill_data.npz`: the design matrix, the true target, and both
teachers' predictions on every row, plus the time index and hemisphere flag so
the search script can reproduce the same split.

    conda activate /glade/work/wchapman/conda-envs/credit-casper-modern
    PYTHONNOUSERSITE=1 python distill_teachers.py
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor

BASE = "/glade/work/wchapman/PYSR_Julio"
SIGMA = float(os.environ.get("GWSRC_SIGMA", 2))
DATA = f"{BASE}/gwsrc_data_s{SIGMA:g}.npz"
OUT = f"{BASE}/distill_data.npz"

# Identical to run_pysr_gwsrc.py. If these drift the comparison is void.
D_REF, P_REF, Y_REF = 3e-7, 3e-8, 4.68e-3
U_REF, Z_REF, Zeta_REF = 10.0, 5000.0, 3e-5
P_FLOOR = 1e-4
TRAIN_FRAC = 0.7
NAMES = ["D", "Dl", "Dg", "P", "Us", "Ul", "Zs", "Zt"]

BATCH = 65536
EPOCHS = 200
PATIENCE = 15


def build(z, tag):
    g = lambda k: z[f"{k}_{tag}"].astype(np.float64)
    cols = {
        "D":   g("tilt_zs") / D_REF,
        "Dl":  g("tilt_zl") / D_REF,
        "Dg":  g("tilt_zg_zs") / D_REF,
        "P":   np.maximum(g("precl") / P_REF, P_FLOOR),
        "Us":  np.hypot(g("u_zs"), g("v_zs")) / U_REF,
        "Ul":  np.hypot(g("u_zl"), g("v_zl")) / U_REF,
        "Zs":  g("z_steer") / Z_REF,
        "Zt":  np.abs(g("zeta_zs")) / Zeta_REF,
    }
    X = np.column_stack([cols[n] for n in NAMES])
    y = np.log(g("epwp_zl") / Y_REF)
    return X, y, z[f"tidx_{tag}"]


def r_log(p, t):
    ok = np.isfinite(p) & np.isfinite(t)
    return float(np.corrcoef(p[ok], t[ok])[0, 1])


class MLP(nn.Module):
    """The nn_gwsrc.py architecture, unchanged."""

    def __init__(self, n_in, hidden=(256, 256, 128), p_drop=0.05):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(p_drop)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(Xtr, ytr, Xva, yva, dev):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    ym, ys = ytr.mean(), ytr.std()

    def prep(X, y=None):
        t = torch.from_numpy(((X - mu) / sd).astype(np.float32))
        return (t,) if y is None else (t, torch.from_numpy(
            ((y - ym) / ys).astype(np.float32)))

    xt, yt = prep(Xtr, ytr)
    xv, yv = prep(Xva, yva)
    xv, yv = xv.to(dev), yv.to(dev)

    net = MLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.MSELoss()

    best, best_state, bad = np.inf, None, 0
    n = xt.shape[0]
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = xt[idx].to(dev), yt[idx].to(dev)
            opt.zero_grad()
            lossf(net(xb), yb).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(xv), yv))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if ep % 20 == 0:
            print(f"    epoch {ep:3d}  val {v:.4f}  best {best:.4f}", flush=True)

    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        out = np.empty(X.shape[0], np.float64)
        CHUNK = 1 << 20
        with torch.no_grad():
            for i in range(0, X.shape[0], CHUNK):
                xb = prep(X[i:i + CHUNK])[0].to(dev)
                out[i:i + CHUNK] = net(xb).cpu().numpy()
        return out * ys + ym

    return predict


def main():
    z = np.load(DATA)
    Xs, ys_, ts = build(z, "sh")
    Xn, yn_, tn = build(z, "nh")

    X = np.concatenate([Xs, Xn])
    y = np.concatenate([ys_, yn_])
    tidx = np.concatenate([ts, tn])
    hemi = np.concatenate([np.zeros(ys_.size, bool), np.ones(yn_.size, bool)])

    tsplit = np.quantile(tidx, TRAIN_FRAC)
    train = tidx <= tsplit
    print(f"{X.shape[0]:,} rows, {X.shape[1]} features   "
          f"{train.sum():,} train / {(~train).sum():,} test (time-blocked)")

    rng = np.random.default_rng(0)
    tr = np.flatnonzero(train)
    if tr.size > 1_000_000:          # the same cap PySR gets
        tr = rng.choice(tr, 1_000_000, replace=False)
    te = np.flatnonzero(~train)
    nh_te = np.flatnonzero((~train) & hemi)

    # --- teacher 1: boosted tree -------------------------------------------
    print("\nfitting tree ...", flush=True)
    tree = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.08, max_leaf_nodes=127,
        early_stopping=True, random_state=0)
    tree.fit(X[tr], y[tr])
    y_tree = tree.predict(X)

    # --- teacher 2: smooth MLP ---------------------------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nfitting mlp on {dev} ...", flush=True)
    va = rng.choice(tr, min(200_000, tr.size), replace=False)
    tr2 = np.setdiff1d(tr, va, assume_unique=False)
    predict = fit_mlp(X[tr2], y[tr2], X[va], y[va], dev)
    y_mlp = predict(X)

    summary = {}
    for name, yp in (("tree", y_tree), ("mlp", y_mlp)):
        summary[name] = dict(
            r_test=r_log(yp[te], y[te]),
            r_nh=r_log(yp[nh_te], y[nh_te]),
            # How jagged the teacher is: correlation between the two teachers
            # tells us how much of what they know is shared.
        )
        print(f"\n{name}: SH+NH test r={summary[name]['r_test']:.4f}   "
              f"NH-only r={summary[name]['r_nh']:.4f}")
    summary["teacher_agreement_r"] = r_log(y_tree, y_mlp)
    print(f"\nteachers agree with each other: r={summary['teacher_agreement_r']:.4f}")
    print("PySR direct fit for reference:   SH 0.6949  NH 0.6979")

    np.savez_compressed(
        OUT + ".tmp.npz", X=X.astype(np.float32), y=y.astype(np.float32),
        y_tree=y_tree.astype(np.float32), y_mlp=y_mlp.astype(np.float32),
        tidx=tidx.astype(np.int16), hemi=hemi, names=np.array(NAMES),
        tsplit=np.float64(tsplit))
    os.replace(OUT + ".tmp.npz", OUT)
    with open(f"{BASE}/distill_teachers.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
