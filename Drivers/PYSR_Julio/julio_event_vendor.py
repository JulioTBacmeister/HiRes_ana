"""
Vendored copies of the four functions needed to rebuild Julio's event-composite
predictor set outside his package.

Why vendor rather than import: `predictors.py` -> `analysis_utils` ->
`event_utils` -> `file_utils` -> `utils.py`, and `utils.py` uses nested-quote
f-strings (PEP 701, Python 3.12+). Julio runs 3.13; the only local env with
torch is 3.11, so the import chain cannot be parsed there. The functions we
actually need are pure numpy, so they are copied verbatim from:

    analysis_utils.py : _is_pressure_like, compute_weighted_centroid_v2,
                        nearest_level, vorticity_centroid_levels
    event_utils.py    : AttrDict, vtime_avg_events
    predictors.py     : dynamic_steer_launch, build_target

Keep in sync if Julio changes them. Checked against the versions dated
2026-08-16 (analysis_utils, event_utils) and 2026-07-30 (predictors).
"""

import numpy as np


# --------------------------------------------------------------- event_utils
class AttrDict(dict):
    """dict with attribute access. The pickles record this class as
    `event_utils.AttrDict`, so a stub module carrying it must be installed in
    sys.modules before pickle.load -- see install_event_utils_stub()."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


def install_event_utils_stub():
    """Make `event_utils.AttrDict` resolvable for unpickling without importing
    Julio's package (which needs Python 3.12+ to parse)."""
    import sys
    import types

    if "event_utils" in sys.modules:
        return
    stub = types.ModuleType("event_utils")
    stub.AttrDict = AttrDict
    sys.modules["event_utils"] = stub


def vtime_avg_events(Eco):
    """Average every *_4D field along the event-internal time axis, keeping the
    axis with length 1."""
    out = {}
    for key, val in Eco.items():
        if key.endswith("4D") and len(np.shape(val)) >= 2:
            out[key] = np.expand_dims(np.mean(val, axis=1), axis=1)
        else:
            out[key] = val
    return AttrDict(out)


# ------------------------------------------------------------ analysis_utils
def _is_pressure_like(coord):
    inc = coord[:, -1] > coord[:, 0]
    if inc.all():
        return True
    if (~inc).all():
        return False
    raise ValueError("mixed coordinate orientation across columns")


def compute_weighted_centroid_v2(w, z, z_min=None, z_max=None):
    w = np.asarray(w, dtype=float)
    z = np.asarray(z, dtype=float)
    if z.shape != w.shape:
        z = np.broadcast_to(z, w.shape)

    desc = z[..., -1] < z[..., 0]
    if np.any(desc):
        m = desc[..., None]
        z = np.where(m, z[..., ::-1], z)
        w = np.where(m, w[..., ::-1], w)

    z1, z2 = z[..., :-1], z[..., 1:]
    w1, w2 = w[..., :-1], w[..., 1:]

    a = np.maximum(z1, z_min) if z_min is not None else z1
    b = np.minimum(z2, z_max) if z_max is not None else z2

    dz = z2 - z1
    valid = (b > a) & (dz > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(dz > 0, (w2 - w1) / np.where(dz > 0, dz, 1.0), 0.0)

    wa = w1 + s * (a - z1)
    wb = w1 + s * (b - z1)
    zm = 0.5 * (a + b)
    wm = w1 + s * (zm - z1)

    seg_w = np.where(valid, 0.5 * (wa + wb) * (b - a), 0.0)
    seg_zw = np.where(valid, (b - a) / 6.0 * (a * wa + 4.0 * zm * wm + b * wb), 0.0)

    denom = seg_w.sum(axis=-1)
    numer = seg_zw.sum(axis=-1)

    out = np.full(denom.shape, np.nan)
    good = denom > 0
    out[good] = numer[good] / denom[good]
    return out


def nearest_level(z_target, zm):
    k = np.full(z_target.shape, -1, dtype=int)
    valid = np.isfinite(z_target)
    if np.any(valid):
        d = np.abs(zm - z_target[..., None])
        k[valid] = np.argmin(d, axis=-1)[valid]
    return k


def vorticity_centroid_levels(vorticity, coord, bound_surface=None, bound_top=None):
    """Steering level = centroid of |vorticity| between the bounds; launch level
    = centroid restricted to levels physically above the steering level."""
    absvort = np.abs(vorticity)
    ncol = absvort.shape[0]

    if _is_pressure_like(coord):
        coord_min, coord_max = bound_top, bound_surface
    else:
        coord_min, coord_max = bound_surface, bound_top

    c_steer = compute_weighted_centroid_v2(absvort, coord,
                                           z_min=coord_min, z_max=coord_max)
    steering_level = nearest_level(c_steer, coord)

    w_top = absvort.copy()
    for i in range(ncol):
        k = steering_level[i]
        if k >= 0:
            w_top[i, k:] = 0.0
        else:
            w_top[i, :] = 0.0

    c_launch = compute_weighted_centroid_v2(w_top, coord)
    launch_level = nearest_level(c_launch, coord)

    return steering_level, launch_level, c_steer, c_launch


# ----------------------------------------------------------------- predictors
def dynamic_steer_launch(Eco=None, zlev=None, use_predictors=None,
                         trange=None, yrange=None, xrange=None,
                         t_avg_flux_pred=False, verbose=False):
    """Verbatim port of predictors.dynamic_steer_launch.

    use_predictors[0] selects the field whose |value| centroid defines the
    steering and launch levels; the rest supply flux-amplitude predictors,
    each sampled four ways (ZS-ZL average, at ZL, at ZS, surface-to-ZS average)
    subject to the skip lists.
    """
    if "abs_zeta_4D" not in Eco:
        Eco["abs_zeta_4D"] = np.abs(Eco.zeta_4D)

    plev = 100_000.0 * np.exp(-zlev / 7000.0)
    nv, nt_v, nz_v, ny_v, nx_v = np.shape(Eco.zeta_4D)
    pmid = np.tile(plev, (nv, 1))

    predictors, predictor_names = [], []

    start_t, stop_t = (0, nt_v) if trange is None else trange
    j0, j1 = (0, ny_v) if yrange is None else yrange
    i0, i1 = (0, nx_v) if xrange is None else xrange

    if use_predictors is None:
        use_predictors = ["vmag_4D", "abs_zeta_4D", "tilt_4D"]

    pred0_n = use_predictors[0]
    if pred0_n == "vmag_4D":
        pred0 = np.sqrt(Eco["u_4D"] ** 2 + Eco["v_4D"] ** 2)
    elif pred0_n == "abs_zeta_4D":
        pred0 = np.abs(Eco["zeta_4D"])
    else:
        pred0 = Eco[pred0_n]

    key_z = f"Dynamic steering and launch based on {pred0_n}"

    zs = np.zeros((nv, nt_v), dtype=int)
    zl = np.zeros((nv, nt_v), dtype=int)
    zoop = np.zeros((nv, nt_v, nz_v))

    for t in np.arange(start_t, stop_t):
        zoop[:, t, :] = np.mean(pred0[:, t, :, j0:j1, i0:i1], axis=(2, 3))
        steering_level, launch_level, _, _ = vorticity_centroid_levels(
            vorticity=zoop[:, t, :], coord=pmid,
            bound_surface=85_000.0, bound_top=5_000.0)
        zs[:, t] = steering_level
        zl[:, t] = launch_level

    skip_ZL = ["thpwp_4D", "stab_4D", "precl_4D"]
    skip_ZSxZL = ["thpwp_4D", "stab_4D", "precl_4D"]
    skip_ZS = [""]
    skip_0xZS = ["precl_4D"]

    # First predictor block: the moving-mountain source wind, |V(ZL) - V(ZS)|.
    for t in np.arange(start_t, stop_t):
        usrc_mm = np.zeros(nv)
        for v in np.arange(nv):
            u_s = np.mean(Eco["u_4D"][v, t, zs[v, t], j0:j1, i0:i1])
            v_s = np.mean(Eco["v_4D"][v, t, zs[v, t], j0:j1, i0:i1])
            u_l = np.mean(Eco["u_4D"][v, t, zl[v, t], j0:j1, i0:i1]) - u_s
            v_l = np.mean(Eco["v_4D"][v, t, zl[v, t], j0:j1, i0:i1]) - v_s
            usrc_mm[v] = np.sqrt(u_l ** 2 + v_l ** 2)
        predictors.append(usrc_mm.copy())
        predictor_names.append(f"U_wv_src_mm t={t - nt_v + 1}, dynamic")

    for pred_ in use_predictors[1:]:
        if not t_avg_flux_pred:
            for t in np.arange(start_t, stop_t):
                for lo_hi, tag, skip in (
                    (lambda v: (zl[v, t], zs[v, t]), "ZS-ZL avg", skip_ZSxZL),
                    (lambda v: (zl[v, t], zl[v, t]), "ZL", skip_ZL),
                    (lambda v: (zs[v, t], zs[v, t]), "ZS", skip_ZS),
                    (lambda v: (zs[v, t], nz_v - 1), "0-ZS avg", skip_0xZS),
                ):
                    if pred_ in skip:
                        continue
                    arr = np.zeros(nv)
                    for v in np.arange(nv):
                        lo, hi = lo_hi(v)
                        arr[v] = np.mean(Eco[pred_][v, t, lo:hi + 1, j0:j1, i0:i1])
                    predictors.append(arr)
                    predictor_names.append(
                        f"avg({pred_}) t={t - nt_v + 1}, dynamic z ({tag})")
        else:
            raise NotImplementedError("t_avg_flux_pred path not vendored")

    if verbose:
        for n, p in zip(predictor_names, predictors):
            print(f"  {n}: min={np.min(p):.4g} max={np.max(p):.4g}")

    return predictors, predictor_names, use_predictors, key_z, "", zs, zl


def build_target(Eco, z_targ=None, targ_scaling=1.0, zl_dyn=None):
    """Event target: epwp averaged over the whole event footprint, at the
    dynamic launch level when one is supplied."""
    if zl_dyn is None:
        return targ_scaling * np.mean(Eco.epwp_4D[:, -1, z_targ, :, :], axis=(2, 1))
    nv = np.shape(Eco.epwp_4D)[0]
    yv = np.zeros(nv)
    for v in np.arange(nv):
        yv[v] = targ_scaling * np.mean(Eco.epwp_4D[v, -1, zl_dyn[v, -1], :, :])
    return yv
