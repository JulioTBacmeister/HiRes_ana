"""
Neural-network reference models for the GW source function.

Purpose is not to ship an NN. It is to answer, per feature set, "how much skill
is available here at all?" -- the role Grundner et al. (2024) give their NNs
before doing equation discovery. That number is what tells you when an equation
has stopped being worth more terms, and it is the honest cap on any distillation.

Three things are done differently from EventCompositeAna_NN-sweep.ipynb, each
targeting a specific weakness visible in that notebook's output:

1. Data.  4.26M SH gridpoints instead of 19,888 composite events. The sweep
   reports train R2 = 0.532 against test R2 = 0.264 -- a gap that size on 15,923
   training samples is a data-starved model, not a saturated one.

2. Loss.  Plain MSE on log tau. The sweep trains on |error|^5, which chases the
   tail at the expense of the bulk. Its own numbers show the cost: test r = 0.704
   implies r^2 = 0.496, but test R2 = 0.398. That ~0.1 gap is not ranking skill,
   it is calibration -- the predictions have the right order and the wrong
   amplitude, and amplitude is what CAM consumes.

3. Split.  Contiguous time blocks, and a separate SH -> NH transfer test. Random
   splits leak across the strong temporal autocorrelation in this field.

Every heavy-tailed positive predictor is log-transformed before standardising,
which is what the sweep does too, and is the right call for a network even
though it would be the wrong call for PySR (see scaling_experiment.py).

    conda activate /glade/work/wchapman/conda-envs/credit-casper-modern
    python nn_gwsrc.py
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "gwsrc_data.npz")
OUT = os.path.join(HERE, "pysr_outputs", "nn_reference.json")
PRED_OUT = os.path.join(HERE, "nn_predictions.npz")

TRAIN_FRAC, VAL_FRAC = 0.6, 0.1     # by time; the last 30% is the test block
BATCH = 65536
MAX_EPOCHS = 200
PATIENCE = 15

# Predictors that span decades and are strictly positive: standardise their logs,
# not themselves, or a handful of extreme columns dominate the first layer.
LOG_VARS = {"tilt_zs", "tilt_zl", "tilt_zs_zl", "tilt_zg_zs", "precl",
            "abs_zeta_zs", "abs_zeta_zl"}

SETS = {
    "D,P only": ["tilt_zs", "precl"],
    "tilt family": ["tilt_zs", "precl", "tilt_zl", "tilt_zs_zl", "tilt_zg_zs"],
    "+winds,vorticity": ["tilt_zs", "precl", "tilt_zl", "tilt_zs_zl",
                         "tilt_zg_zs", "spd_zs", "spd_zl", "shear",
                         "abs_zeta_zs", "abs_zeta_zl"],
    "everything": ["tilt_zs", "precl", "tilt_zl", "tilt_zs_zl", "tilt_zg_zs",
                   "spd_zs", "spd_zl", "shear", "abs_zeta_zs", "abs_zeta_zl",
                   "z_steer", "z_launch", "dz", "u_zs", "u_zl"],
    # The eight PySR is searching over, so the equation has a like-for-like cap.
    "pysr eight": ["tilt_zs", "tilt_zl", "tilt_zg_zs", "precl", "spd_zs",
                   "spd_zl", "z_steer", "abs_zeta_zs"],
}


def candidates(hemi):
    z = np.load(DATA)
    g = lambda k: z[f"{k}_{hemi}"].astype(np.float32)
    u_zs, v_zs, u_zl, v_zl = g("u_zs"), g("v_zs"), g("u_zl"), g("v_zl")
    c = {
        "tilt_zs": g("tilt_zs"), "tilt_zl": g("tilt_zl"),
        "tilt_zs_zl": g("tilt_zs_zl"), "tilt_zg_zs": g("tilt_zg_zs"),
        "precl": g("precl"),
        "abs_zeta_zs": np.abs(g("zeta_zs")), "abs_zeta_zl": np.abs(g("zeta_zl")),
        "z_steer": g("z_steer"), "z_launch": g("z_launch"),
        "dz": g("dz_launch_steer"),
        "spd_zs": np.hypot(u_zs, v_zs), "spd_zl": np.hypot(u_zl, v_zl),
        "shear": np.hypot(u_zl - u_zs, v_zl - v_zs),
        "u_zs": u_zs, "u_zl": u_zl,
    }
    y = np.log(g("epwp_zl").astype(np.float64)).astype(np.float32)
    return c, y, z[f"tidx_{hemi}"]


def design(cand, keys):
    cols = []
    for k in keys:
        v = cand[k].astype(np.float64)
        # The floor matters: precl reaches 1e-27 and its log would otherwise
        # swamp the standardisation.
        cols.append(np.log(np.maximum(v, 1e-14)) if k in LOG_VARS else v)
    return np.column_stack(cols).astype(np.float32)


class MLP(nn.Module):
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


def metrics(pred, true):
    """
    r, R2, and R2 after an affine recalibration.

    The third one separates two very different failures. If a model transfers to
    the other hemisphere with a bad R2 but recovers once you allow a shift and a
    scale in log space, the *function* transferred and only its calibration did
    not -- fixable with two numbers. If R2_cal stays bad, the learned response is
    itself wrong there and no amount of retuning saves it.
    """
    # float64 throughout. np.polyfit on ~1e6 float32 values is badly conditioned
    # and silently returns a wrong slope (0.50 where the answer is 0.95), which
    # then propagates into a nonsensical R2_cal below R2. Compute the fit in
    # closed form instead of trusting a Vandermonde solve at this length.
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    r = float(np.corrcoef(pred, true)[0, 1])
    denom = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1.0 - np.sum((pred - true) ** 2) / denom)

    pm, tm = pred.mean(), true.mean()
    slope = float(np.sum((pred - pm) * (true - tm)) / np.sum((pred - pm) ** 2))
    offset = float(tm - slope * pm)
    r2_cal = float(1.0 - np.sum((slope * pred + offset - true) ** 2) / denom)
    # Best affine fit, so this is r^2 by construction -- a cheap invariant that
    # would have caught the float32 failure immediately.
    assert abs(r2_cal - r * r) < 1e-6, f"R2_cal {r2_cal} != r^2 {r * r}"
    return r, r2, r2_cal, slope, offset


def run(name, keys, sh, nh, dev, train_on="sh"):
    """
    train_on='sh'   : fit on SH only, score NH as a pure transfer test.
    train_on='both' : fit on SH and NH together, split on time. Every SH-trained
                      model above transfers to the NH with a slope near 0.45 --
                      it overpredicts the NH dynamic range roughly two-fold --
                      and that is a single global miscalibration, not five
                      independent ones. Training on both hemispheres is the
                      direct fix, and it also turns the feature-set question into
                      the one that matters for a global parameterization: does
                      this predictor earn its place on unseen *times*, in both
                      hemispheres at once?
    """
    cand, y, tidx = sh
    X = design(cand, keys)

    if train_on == "both":
        cand_n, y_n, tidx_n = nh
        X = np.concatenate([X, design(cand_n, keys)])
        y = np.concatenate([y, y_n])
        # Keep the hemisphere label so the held-out block can be scored
        # separately -- a global average would hide exactly the asymmetry the
        # SH-only runs exposed.
        hemi = np.concatenate([np.zeros(tidx.size, bool), np.ones(tidx_n.size, bool)])
        tidx = np.concatenate([tidx, tidx_n])
    else:
        hemi = np.zeros(tidx.size, bool)

    t_tr = np.quantile(tidx, TRAIN_FRAC)
    t_va = np.quantile(tidx, TRAIN_FRAC + VAL_FRAC)
    tr, va, te = tidx <= t_tr, (tidx > t_tr) & (tidx <= t_va), tidx > t_va

    # Standardise on the training block only.
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    ymu, ysd = y[tr].mean(), y[tr].std()

    to = lambda a: torch.as_tensor(a, device=dev)
    Xt = to((X - mu) / sd)
    yt = to((y - ymu) / ysd)
    itr, iva, ite = (to(np.flatnonzero(m).astype(np.int64)) for m in (tr, va, te))

    torch.manual_seed(0)
    model = MLP(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    lossf = nn.MSELoss()

    best, best_state, bad = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = itr[torch.randperm(itr.numel(), device=dev)]
        for i in range(0, perm.numel(), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            lossf(model(Xt[b]), yt[b]).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xt[iva]), yt[iva]).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if epoch % 20 == 0:
            print(f"    epoch {epoch:3d}  val {vl:.4f}  lr {sched.get_last_lr()[0]:.2e}",
                  flush=True)

    model.load_state_dict(best_state)
    model.eval()

    CHUNK = 1 << 20

    def predict(Xa):
        out = []
        with torch.no_grad():
            for i in range(0, Xa.shape[0], CHUNK):
                out.append(model(Xa[i:i + CHUNK]).cpu().numpy())
        return np.concatenate(out) * ysd + ymu

    res = {"n_features": len(keys), "epochs": epoch + 1, "train_on": train_on}
    p_te = predict(Xt[ite])
    (res["r_test"], res["R2_test"], res["R2_test_cal"],
     res["slope_test"], res["off_test"]) = metrics(p_te, y[te])

    if train_on == "both":
        # Held-out times, scored per hemisphere. Both are out-of-sample in time;
        # neither is out-of-sample in space any more, which is the point.
        for tag, m in (("sh", ~hemi[te]), ("nh", hemi[te])):
            (res[f"r_{tag}"], res[f"R2_{tag}"], res[f"R2_{tag}_cal"],
             res[f"slope_{tag}"], res[f"off_{tag}"]) = metrics(p_te[m], y[te][m])
        print(f"  {name:18s} n={len(keys):2d}  SH r={res['r_sh']:.4f} "
              f"R2={res['R2_sh']:+.4f}   NH r={res['r_nh']:.4f} "
              f"R2={res['R2_nh']:+.4f} (slope {res['slope_nh']:.2f})   "
              f"({epoch + 1} epochs)", flush=True)
    else:
        # SH -> NH transfer, standardised with the SH statistics: the point is
        # whether the SH-trained function survives, not whether it can be refitted.
        cand_n, y_n, _ = nh
        Xn = (design(cand_n, keys) - mu) / sd
        p_nh = predict(to(Xn.astype(np.float32)))
        (res["r_nh"], res["R2_nh"], res["R2_nh_cal"],
         res["slope_nh"], res["off_nh"]) = metrics(p_nh, y_n)
        print(f"  {name:18s} n={len(keys):2d}  test r={res['r_test']:.4f} "
              f"R2={res['R2_test']:.4f}   NH r={res['r_nh']:.4f} "
              f"R2={res['R2_nh']:+.4f} R2cal={res['R2_nh_cal']:+.4f} "
              f"(slope {res['slope_nh']:.2f})   ({epoch + 1} epochs)", flush=True)
    return res, (p_te, y[te], np.flatnonzero(te))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}")

    sh = candidates("sh")
    nh = candidates("nh")
    print(f"SH {sh[1].size:,} samples   NH {nh[1].size:,} samples")

    out, saved = {}, {}
    for train_on in ("sh", "both"):
        print(f"\n{'=' * 78}\ntrained on: {train_on}\n{'=' * 78}", flush=True)
        for name, keys in SETS.items():
            t0 = time.time()
            res, preds = run(name, keys, sh, nh, dev, train_on=train_on)
            res["wall_s"] = time.time() - t0
            out[f"{name} [{train_on}]"] = res
            if name == "pysr eight" and train_on == "sh":
                saved = dict(pred_test=preds[0], true_test=preds[1],
                             idx_test=preds[2])

    for train_on, title in (("sh", "TRAINED ON SH ONLY (NH is a pure transfer test)"),
                            ("both", "TRAINED ON BOTH (NH is held-out times only)")):
        print(f"\n{title}")
        print(f"{'feature set':18s} {'n':>3} {'SH r':>7} {'SH R2':>8} "
              f"{'NH r':>7} {'NH R2':>8} {'NH R2cal':>9} {'NH slope':>9}")
        for k, v in out.items():
            if v["train_on"] != train_on:
                continue
            sh_r = v["r_sh"] if train_on == "both" else v["r_test"]
            sh_r2 = v["R2_sh"] if train_on == "both" else v["R2_test"]
            print(f"{k.split(' [')[0]:18s} {v['n_features']:3d} {sh_r:7.3f} "
                  f"{sh_r2:+8.3f} {v['r_nh']:7.3f} {v['R2_nh']:+8.3f} "
                  f"{v['R2_nh_cal']:+9.3f} {v['slope_nh']:9.2f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    if saved:
        np.savez_compressed(PRED_OUT, **saved)
        print(f"wrote {PRED_OUT}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
