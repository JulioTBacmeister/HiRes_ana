"""
Put the event-composite NN and the gridded models on one axis.

The two frameworks report the same statistic (log-space Pearson r) against
different targets, so their scores are not comparable as published. This joins
them: every event-framework score is multiplied by sqrt(variance_ratio) from
event_ceiling.py to give the r it would correspond to if the same skill were
applied column by column, which is what CAM's parameterization has to do.

    conda activate npl-2026a && PYTHONNOUSERSITE=1 python event_vs_grid.py
"""

import glob
import json
import os

BASE = "/glade/work/wchapman/PYSR_Julio"


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def gridded_rows():
    """Elbow of each PySR front: the largest r_test per run, with its NH score."""
    rows = []
    for path in sorted(glob.glob(f"{BASE}/pysr_outputs/front_*.json")):
        tag = os.path.basename(path)[len("front_"):-len(".json")]
        if tag.endswith("_prelim"):
            continue
        d = load(path)
        best = max(d["rows"], key=lambda r: r.get("r_test", -9))
        rows.append({
            "model": f"PySR {tag}",
            "framework": "gridded",
            "fit_on": d.get("train_on", "sh"),
            "detail": f"cx={best['complexity']}",
            "sh": best.get("r_test"),
            "nh": best.get("r_nh"),
        })
    refs = load(f"{BASE}/pysr_outputs/front_dp.json", {}).get("refs", {})
    if "julio_r" in refs:
        rows.append({"model": "Julio hand-built form", "framework": "gridded",
                     "fit_on": "sh", "detail": "7 params",
                     "sh": refs["julio_r"], "nh": refs.get("julio_r_nh")})
    return rows


def event_rows(nn):
    out = []
    for r in nn:
        loss = "MSE" if r["loss_power"] == 2 else f"|e|^{r['loss_power']}"
        out.append({
            "model": f"NN {r['set']}",
            "framework": "event",
            "fit_on": r["train_on"],
            "detail": f"{r['n_predictors']} preds, {loss}",
            "sh": r["test"]["r"],
            "nh": r["nh"]["r"],
            "sh_slope": r["test"]["slope"],
            "nh_slope": r["nh"]["slope"],
        })
    return out


def main():
    nn = load(f"{BASE}/event_nn_results.json")
    ceil = load(f"{BASE}/event_ceiling.json")
    if nn is None or ceil is None:
        raise SystemExit("run nn_event_gwsrc.py and event_ceiling.py first")

    k_sh = ceil["hemispheres"]["sh"]["variance_ratio"] ** 0.5
    k_nh = ceil["hemispheres"]["nh"]["variance_ratio"] ** 0.5

    rows = event_rows(nn) + gridded_rows()

    def cell(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    print(f"\nFootprint-average correction: SH x{k_sh:.3f}, NH x{k_nh:.3f}")
    print("(an event score times this is the pointwise r it implies)\n")

    hdr = ("| model | frame | fit | detail | SH r | NH r "
           "| SH r (pointwise) | NH r (pointwise) |")
    print(hdr)
    print("|" + "---|" * 8)
    for r in sorted(rows, key=lambda x: -(x["nh"] or 0)):
        ksh, knh = (k_sh, k_nh) if r["framework"] == "event" else (1.0, 1.0)
        psh = r["sh"] * ksh if isinstance(r["sh"], float) else None
        pnh = r["nh"] * knh if isinstance(r["nh"], float) else None
        print(f"| {r['model']} | {r['framework']} | {r['fit_on']} | "
              f"{r['detail']} | {cell(r['sh'])} | {cell(r['nh'])} | "
              f"{cell(psh)} | {cell(pnh)} |")

    print("\nCalibration slopes on the event models (1.0 = correctly dispersed):")
    for r in rows:
        if r["framework"] == "event":
            print(f"  {r['model']:<32s} {r['detail']:<22s} "
                  f"SH {r['sh_slope']:.2f}   NH {r['nh_slope']:.2f}")


if __name__ == "__main__":
    main()
