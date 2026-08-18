import numpy as np

def bin_shape_1d(d, p, y, n_d=16, n_p=12, min_count=200,
                 y_floor=None, p_dry=1e-12):
    good = np.isfinite(d) & np.isfinite(p) & np.isfinite(y)
    if y_floor is not None:
        good &= y > y_floor
    d, p, y = d[good], p[good], y[good]
    print(f"{good.sum()} points ({(p <= p_dry).sum()} dry)")

    d_edges = np.quantile(d, np.linspace(0, 1, n_d + 1))
    wet = p > p_dry
    p_edges = np.concatenate([[0.0, p_dry],
                              np.quantile(p[wet], np.linspace(1/n_p, 1, n_p))])

    di = np.clip(np.digitize(d, d_edges) - 1, 0, len(d_edges) - 2)
    pi = np.clip(np.digitize(p, p_edges) - 1, 0, len(p_edges) - 2)

    shape = (len(d_edges) - 1, len(p_edges) - 1)
    flat = di * shape[1] + pi
    cnt = np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape)

    med = np.full(shape, np.nan)
    order = np.argsort(flat)
    fs, ys = flat[order], y[order]
    starts = np.searchsorted(fs, np.arange(cnt.size))
    ends = np.searchsorted(fs, np.arange(cnt.size), side='right')
    for k in np.flatnonzero(cnt.ravel() >= min_count):
        med.ravel()[k] = np.median(ys[starts[k]:ends[k]])

    return med, cnt, d_edges, p_edges


def cell_spread(d, p, y, d_edges, p_edges, min_count=200, p_dry=1e-12):
    """
    Within-cell spread and variance decomposition on the same bins
    as bin_shape_1d. Returns q16, q84, spread ratio, and the implied
    ceiling on log-space correlation for any f(D, P).
    """
    good = np.isfinite(d) & np.isfinite(p) & np.isfinite(y) & (y > 0)
    d, p, y = d[good], p[good], y[good]
    ly = np.log(y)

    di = np.clip(np.digitize(d, d_edges) - 1, 0, len(d_edges) - 2)
    pi = np.clip(np.digitize(p, p_edges) - 1, 0, len(p_edges) - 2)
    shape = (len(d_edges) - 1, len(p_edges) - 1)
    flat = di * shape[1] + pi

    order = np.argsort(flat, kind='stable')
    fs, lys = flat[order], ly[order]
    starts = np.searchsorted(fs, np.arange(shape[0] * shape[1]))
    ends   = np.searchsorted(fs, np.arange(shape[0] * shape[1]), side='right')
    cnt    = (ends - starts).reshape(shape)

    q16 = np.full(shape, np.nan)
    q84 = np.full(shape, np.nan)
    cell_mean = np.full(shape, np.nan)
    cell_var  = np.full(shape, np.nan)

    for k in np.flatnonzero(cnt.ravel() >= min_count):
        seg = lys[starts[k]:ends[k]]
        q16.ravel()[k], q84.ravel()[k] = np.percentile(seg, [16, 84])
        cell_mean.ravel()[k] = seg.mean()
        cell_var.ravel()[k]  = seg.var()

    ok = np.isfinite(cell_mean)
    w  = cnt[ok].astype(float)
    w /= w.sum()

    var_within  = np.sum(w * cell_var[ok])
    grand_mean  = np.sum(w * cell_mean[ok])
    var_between = np.sum(w * (cell_mean[ok] - grand_mean) ** 2)
    r_ceiling   = np.sqrt(var_between / (var_between + var_within))

    print(f"cells used:          {ok.sum()} of {ok.size}")
    print(f"within-cell  var:    {var_within:.4f}  (log units)")
    print(f"between-cell var:    {var_between:.4f}")
    print(f"explained fraction:  {var_between/(var_between+var_within):.3f}")
    print(f"ceiling on log-r:    {r_ceiling:.3f}")
    print(f"median 16–84 spread: {np.nanmedian(np.exp(q84-q16)):.2f}x")

    return dict(q16=q16, q84=q84, cnt=cnt, cell_mean=cell_mean,
                cell_var=cell_var, r_ceiling=r_ceiling,
                var_within=var_within, var_between=var_between)

def ema(X, alpha):
    out = np.empty_like(X)
    out[0] = X[0]
    for t in range(1, X.shape[0]):
        out[t] = alpha * out[t-1] + (1 - alpha) * X[t]
    return out