"""
Distillation search: fit PySR to a teacher's predictions, score it on real data.

The premise, from form_vs_ceiling.py: on the `full` 8-feature set a boosted tree
reaches 0.730/0.735 where the best direct-fit equation reaches 0.695/0.698. The
search is fitting a target that is ~55% irreducible scatter at sigma=2, so the
fitness signal separating two candidate expressions is weak. Handing it a
teacher's predictions removes the scatter entirely and leaves a smooth
deterministic function to describe.

The rule that keeps this honest: **the equation is always scored against the
real data**, on held-out times and on the NH. Fidelity to the teacher is
reported too, but it is a diagnostic, never the result. A distilled equation
that tracks its teacher perfectly and loses in the NH has learned the teacher's
overfitting, which is exactly the failure mode this is testing for.

Usage:
    conda activate /glade/work/wchapman/conda-envs/L96M2lines
    python run_pysr_distill.py {tree|mlp|data} [niterations]

`data` re-runs the direct fit through this same script, as a control -- same
budget, same split, so the three numbers are comparable.
"""

import json
import os
import sys
import time

import numpy as np
from pysr import PySRRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "distill_data.npz")
OUTDIR = os.path.join(HERE, "pysr_outputs")


def r_log(p, t):
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 2:
        return float("nan")
    return float(np.corrcoef(p[ok], t[ok])[0, 1])


def main():
    teacher = sys.argv[1] if len(sys.argv) > 1 else "tree"
    niter = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    assert teacher in ("tree", "mlp", "data")
    tag = f"distill_{teacher}" + os.environ.get("RUN_TAG", "")

    z = np.load(DATA)
    X, y = z["X"].astype(np.float64), z["y"].astype(np.float64)
    names = [str(s) for s in z["names"]]
    tidx, hemi = z["tidx"], z["hemi"]
    train = tidx <= z["tsplit"]

    # What the search fits. `y` is the real (noisy) target; the teachers are
    # smooth functions of X that already know most of what is learnable.
    target = {"data": y, "tree": z["y_tree"].astype(np.float64),
              "mlp": z["y_mlp"].astype(np.float64)}[teacher]

    nproc = int(os.environ.get("PYSR_PROCS", 0)) or len(os.sched_getaffinity(0))
    print(f"teacher={teacher}  niterations={niter}  procs={nproc}")
    print(f"{X.shape[0]:,} rows, {len(names)} features {names}")
    print(f"    {train.sum():,} train / {(~train).sum():,} test (time-blocked)")
    if teacher != "data":
        print(f"    teacher r vs real data (all rows): "
              f"{r_log(target, y):.4f}", flush=True)

    rng = np.random.default_rng(0)
    tr = np.flatnonzero(train)
    if tr.size > 1_000_000:
        tr = rng.choice(tr, 1_000_000, replace=False)

    os.makedirs(OUTDIR, exist_ok=True)
    model = PySRRegressor(
        niterations=niter,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["exp", "log", "sqrt", "square"],
        populations=3 * nproc,
        population_size=50,
        maxsize=40,
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
        timeout_in_seconds=int(os.environ.get("PYSR_TIMEOUT", 5 * 3600)),
        output_directory=OUTDIR,
        run_id=f"gwsrc_{tag}",
        progress=False,
        verbosity=1,
    )

    t0 = time.time()
    model.fit(X[tr], target[tr], variable_names=names)
    print(f"search wall time: {(time.time() - t0) / 60:.1f} min", flush=True)

    sh_te = (~train) & (~hemi)
    nh_te = (~train) & hemi

    rows = []
    for i in range(len(model.equations_)):
        eq = model.equations_.iloc[i]
        try:
            p_sh = model.predict(X[sh_te], index=i)
            p_nh = model.predict(X[nh_te], index=i)
            p_tr = model.predict(X[train], index=i)
        except Exception as e:                        # noqa: BLE001
            print(f"  eq {i} failed to evaluate: {e}")
            continue
        rows.append(dict(
            index=i, complexity=int(eq["complexity"]), loss=float(eq["loss"]),
            equation=str(eq["equation"]),
            # Scored against REAL data, always.
            r_train=r_log(p_tr, y[train]),
            r_test=r_log(p_sh, y[sh_te]),
            r_nh=r_log(p_nh, y[nh_te]),
            # Diagnostic only: how faithfully it reproduces its teacher.
            r_teacher=r_log(p_sh, target[sh_te]),
        ))

    print(f"\n{'cx':>3} {'r_train':>8} {'r_sh':>8} {'r_nh':>8} {'r_teach':>8}  equation")
    for r in rows:
        print(f"{r['complexity']:3d} {r['r_train']:8.4f} {r['r_test']:8.4f} "
              f"{r['r_nh']:8.4f} {r['r_teacher']:8.4f}  {r['equation'][:90]}")

    best = max(rows, key=lambda r: r["r_test"]) if rows else None
    if best:
        print(f"\nbest vs real data: cx={best['complexity']}  "
              f"SH={best['r_test']:.4f}  NH={best['r_nh']:.4f}")
        print("direct-fit reference:            SH 0.6949  NH 0.6979")
        print("tree teacher:                    SH 0.7304  NH 0.7349")

    with open(os.path.join(OUTDIR, f"front_{tag}.json"), "w") as f:
        json.dump(dict(mode=tag, teacher=teacher, names=names, rows=rows,
                       train_on="both", sigma=2.0), f, indent=1)
    print(f"\nwrote {OUTDIR}/front_{tag}.json")


if __name__ == "__main__":
    sys.exit(main())
