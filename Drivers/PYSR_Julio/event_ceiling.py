"""
Translate event-composite skill into gridded skill.

Julio's NN scores ~0.77 log-space r; our best gridded model scores ~0.70, and
the hand-built form 0.578. Those numbers are not comparable, because they are
not predicting the same thing:

  gridded  : log(epwp) at one column, one timestep      -- 1.3e6 samples
  event    : log(mean epwp over an 11x11 footprint,     -- 7.6e3 samples
             averaged over 3 internal timesteps), at
             the event's dynamic launch level

Averaging over 121 columns is exactly the operation that removes the
within-cell scatter that caps the gridded problem at r ~ 0.583. This script
measures how much variance that averaging removes, which converts one score
into the other:

    r_pointwise_equivalent = r_event * sqrt( var[log(footprint mean)]
                                             / var[log(pointwise)] )

Run in any env with numpy (the .npz is plain arrays):
    conda activate npl-2026a && PYTHONNOUSERSITE=1 python event_ceiling.py
"""

import json

import numpy as np

DATA = "/glade/work/wchapman/PYSR_Julio/event_data.npz"
OUT = "/glade/work/wchapman/PYSR_Julio/event_ceiling.json"

# Log-space r reported for the event framework. Overwritten from
# event_nn_results.json when that exists.
R_EVENT_DEFAULT = 0.77


def main():
    d = np.load(DATA, allow_pickle=True)

    try:
        with open("/glade/work/wchapman/PYSR_Julio/event_nn_results.json") as f:
            res = json.load(f)
        # Julio's published configuration, so the translated number is the one
        # he quotes rather than the best we could coax out of his architecture.
        best = next(r for r in res
                    if r["set"] == "tilt_levs+tilt+precl"
                    and r["loss_power"] == 5 and r["train_on"] == "sh")
        r_event = best["test"]["r"]
        label = f"{best['set']}, |e|^5, SH-fit, SH-test"
    except (FileNotFoundError, ValueError, KeyError):
        r_event, label = R_EVENT_DEFAULT, "Julio's quoted value"

    out = {"r_event": r_event, "r_event_source": label, "hemispheres": {}}
    print(f"event-framework r = {r_event:.3f}  [{label}]\n")

    for tag in ("sh", "nh"):
        foot = d[f"foot_{tag}"].astype(np.float64)      # (nv, ny, nx)
        nv, ny, nx = foot.shape

        # Drop events with any non-positive footprint value: log is undefined
        # there and epwp = sqrt(u'w'^2 + v'w'^2) should never be zero.
        good = np.all(foot > 0, axis=(1, 2))
        foot = foot[good]

        lp = np.log(foot).ravel()                        # pointwise log epwp
        lm = np.log(foot.mean(axis=(1, 2)))              # the event target

        var_point = float(lp.var())
        var_mean = float(lm.var())
        ratio = var_mean / var_point

        # What fraction of pointwise variance survives the footprint average --
        # i.e. the most a perfect event-level model could explain pointwise.
        r_equiv = r_event * np.sqrt(ratio)

        # Within-footprint spread, as a spread factor rather than a variance.
        within = float(np.mean(np.log(foot).var(axis=(1, 2))))

        rec = dict(
            n_events=int(foot.shape[0]),
            n_dropped=int((~good).sum()),
            footprint=[int(ny), int(nx)],
            var_pointwise=var_point,
            var_footprint_mean=var_mean,
            var_within_footprint=within,
            variance_ratio=ratio,
            r_pointwise_equivalent=float(r_equiv),
            spread_factor_1sigma=float(np.exp(np.sqrt(within))),
        )
        out["hemispheres"][tag] = rec

        print(f"[{tag.upper()}]  {rec['n_events']} events, {ny}x{nx} footprint"
              + (f"  ({rec['n_dropped']} dropped)" if rec["n_dropped"] else ""))
        print(f"  var log(epwp) pointwise        : {var_point:.4f}")
        print(f"  var log(epwp) footprint mean   : {var_mean:.4f}")
        print(f"  mean within-footprint variance : {within:.4f}"
              f"   (1-sigma spread factor {rec['spread_factor_1sigma']:.2f}x)")
        print(f"  variance surviving the average : {ratio:.3f}")
        print(f"  => r={r_event:.3f} on the event target is worth "
              f"r={r_equiv:.3f} pointwise\n")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
