"""
source_fit.py
=============
Stage 1 fitting for the GW source function.

    tau = y0 + A * D^a / (1 + D/D0) + B * D^c * P^b

Seven parameters: y0, A, a, D0, B, b, c.  Setting c = 0 recovers the
six-parameter additive form, so the two models are fitted by the same
code and are directly comparable.

The convective term retains the factor P^b, so it still vanishes for
dry columns.  The dry class therefore continues to identify
(y0, A, a, D0) on its own, which was the original argument for an
additive rather than a purely multiplicative form.

At fixed (a, D0, b, c) the model is linear in (y0, A, B).  The grid
stage exploits this: it grids only the four exponent-like parameters
and solves for the three coefficients by non-negative least squares.

Companion to distro_fitting.bin_shape_1d / cell_spread.
"""

import numpy as np
from scipy.optimize import nnls, least_squares


PARAM_KEYS = ['y0', 'A', 'a', 'D0', 'B', 'b', 'c']

# Reference D for the modulation factor (d/D_REF)^c.  Without this
# normalisation B absorbs a factor of D_REF^-c and changes by orders of
# magnitude between the c = 0 and c > 0 fits, which makes the two
# models impossible to compare by eye and gives the polish stage a
# badly scaled starting point.  Set near the middle of the D range.
D_REF = 3e-7

# Reference P for the convective term, chosen near the geometric centre
# of the wet-event range so B = "convective coefficient at a typical
# wet P", analogous to D_REF for A. See source_model docstring.
P_REF = 3e-8


# =============================================================================
# MODEL
# =============================================================================

def source_model(d, p, y0, A, a, D0, B, b, c=0.0, mult_A=1.0, mult_B=1.0):
    """
    Evaluate the source function.

        tau = y0 + mult_A * A * (D/D_REF)^a / (1 + D/D0)
                 + mult_B * B * (P/D_REF)^b... 

    c = 0 gives the purely additive six-parameter form.
    c > 0 lets the dynamical state modulate the convective term.

    Both A and B are defined as the *value of their term at the
    reference point* (D = D_REF for the dynamical term; D = D_REF,
    P = P_REF for the convective term), not as raw power-law
    prefactors.  That fixes two things that raw prefactors get wrong:

    1. Comparability.  A and B are both then in flux units, evaluated
       at a shared reference condition, so A vs B is a meaningful
       statement about which mechanism dominates -- unlike raw
       power-law coefficients, which carry different units (flux per
       D^a vs flux per P^b) and are not comparable as numbers.

    2. Tuning stability.  a and D0 trade off against each other along
       a near-degenerate ridge (see fit diagnostics) -- two fits with
       very different (A, a, D0) can describe the same curve. Pinning
       A to its value at D_REF keeps that number stable across refits
       and grid resolutions, so "tune A by 20%" means the same thing
       run to run. Without the pin, A's baseline itself drifts with
       the ridge and a fixed multiplicative tuning factor has no
       consistent physical meaning.

    mult_A, mult_B : independent tuning knobs, default 1.0. Multiply
    each term's contribution uniformly at all (D, P) without altering
    its shape (a, D0, b, c). These are the only two parameters a CAM
    tuner should touch -- everything else comes from the fit.
    """
    dyn = mult_A * A * (d / D_REF)**a / (1.0 + d / D0)
    conv = mult_B * B * (p / P_REF)**b * ((d / D_REF)**c if c else 1.0)
    return y0 + dyn + conv


def _design(d, p, a, D0, b, c, d_pow_c=None):
    """
    Design matrix for the linear-in-coefficients sub-problem.

    Columns: [1, (D/D_REF)^a/(1+D/D0), (D/D_REF)^c * (P/P_REF)^b]
    Coefficients: [y0, A, B]  -- A, B are term values at the
    reference point (D_REF[, P_REF]), see source_model docstring.

    d_pow_c : precomputed (d/D_REF)**c, optional (grid stage hoists this).
    """
    if d_pow_c is None:
        d_pow_c = (d / D_REF)**c if c else 1.0
    return np.column_stack([
        np.ones_like(d),
        (d / D_REF)**a / (1.0 + d / D0),
        (p / P_REF)**b * d_pow_c,
    ])


# =============================================================================
# STAGE 1a -- NNLS GRID SEARCH
# =============================================================================

def fit_nnls_grid(d, p, y, a_grid=None, b_grid=None, D0_grid=None,
                  c_grid=None, use_y0=True, sample=100_000, seed=0,
                  verbose=True):
    """
    Grid over (a, D0, b, c); solve for (y0, A, B) by NNLS at each node.

    Non-negativity is a diagnostic, not a convenience: a fit that wants
    B < 0 means the functional form is wrong, not that precipitation
    suppresses wave generation.

    Parameters
    ----------
    d, p, y : 1D arrays, masked and raveled
    c_grid : 1D array or None
        Include 0.0 so the six-parameter model lies inside the search
        space.  Pass c_grid=[0.0] to fit the six-parameter form only.
    use_y0 : bool
        If False, drop the constant background term: the fit is forced
        through zero, so the scheme produces no flux where there is
        neither dynamical forcing nor precipitation.  Costs some skill
        at the low-D end, where the dry curve flattens toward a
        non-zero floor, but avoids committing to a background flux.
    sample : int or None
        Subsample size for the grid stage.  The exponents are well
        constrained by far fewer points than the polish stage needs.

    Returns
    -------
    dict with keys: params, resid_cube, coef_cube, and the four grids.
    resid_cube has shape (n_a, n_D0, n_b, n_c) -- inspect it to see how
    shallow the minimum is, which matters given D-P collinearity.
    """
    if a_grid is None:
        a_grid = np.linspace(0.1, 2.5, 25)
    if b_grid is None:
        b_grid = np.linspace(0.1, 1.5, 15)
    if D0_grid is None:
        D0_grid = np.logspace(np.log10(3e-8), np.log10(3e-6), 11)
    if c_grid is None:
        c_grid = np.linspace(0.0, 1.0, 11)

    a_grid, b_grid = np.asarray(a_grid, float), np.asarray(b_grid, float)
    D0_grid, c_grid = np.asarray(D0_grid, float), np.asarray(c_grid, float)

    if sample is not None and sample < d.size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(d.size, size=sample, replace=False)
        ds, ps, ys = d[idx], p[idx], y[idx]
    else:
        ds, ps, ys = d, p, y

    shape = (len(a_grid), len(D0_grid), len(b_grid), len(c_grid))
    resid_cube = np.full(shape, np.inf)
    coef_cube = np.zeros(shape + (3,))

    # hoist the expensive powers out of the inner loops
    ones = np.ones_like(ds)
    d_pow_c = [(ds / D_REF)**cc if cc else ones for cc in c_grid]
    p_pow_b = [(ps / P_REF)**bb for bb in b_grid]

    if verbose:
        print(f"NNLS grid: {int(np.prod(shape))} nodes on {ds.size} points")

    for i, a in enumerate(a_grid):
        da = (ds / D_REF)**a
        for j, D0 in enumerate(D0_grid):
            dyn = da / (1.0 + ds / D0)
            for k in range(len(b_grid)):
                pb = p_pow_b[k]
                for l in range(len(c_grid)):
                    cols = [dyn, pb * d_pow_c[l]]
                    if use_y0:
                        cols = [ones] + cols
                    coef, r = nnls(np.column_stack(cols), ys)
                    resid_cube[i, j, k, l] = r
                    coef_cube[i, j, k, l] = coef if use_y0 \
                        else np.concatenate([[0.0], coef])

    i, j, k, l = np.unravel_index(np.argmin(resid_cube), shape)
    y0, A, B = coef_cube[i, j, k, l]
    par = dict(y0=y0, A=A, a=a_grid[i], D0=D0_grid[j],
               B=B, b=b_grid[k], c=c_grid[l])

    if verbose:
        print("NNLS grid best fit:")
        for key in PARAM_KEYS:
            print(f"  {key:3s} = {par[key]:.6g}")
        print(f"  residual = {resid_cube[i, j, k, l]:.6g}")
        if B <= 0:
            print("  WARNING: B pinned at zero -- convective term not "
                  "supported by this sample")
        for idx, grid, name in [(i, a_grid, 'a'), (j, D0_grid, 'D0'),
                                (k, b_grid, 'b'), (l, c_grid, 'c')]:
            if len(grid) > 1 and idx in (0, len(grid) - 1):
                print(f"  WARNING: {name} optimum on grid edge "
                      f"({grid[idx]:.6g}) -- widen the grid")

    return dict(params=par, resid_cube=resid_cube, coef_cube=coef_cube,
                a_grid=a_grid, b_grid=b_grid, D0_grid=D0_grid,
                c_grid=c_grid)


# =============================================================================
# STAGE 1b -- LOG-SPACE POLISH
# =============================================================================

def fit_log_polish(d, p, y, par0, fix_c=False, use_y0=None,
                   bounds=None, verbose=True):
    """
    Refine by minimising log(model) - log(y), matching the log-space
    scoring used for the MLP. NNLS minimises absolute residual and is
    dominated by the largest events; this weights all decades equally.

    Parameters are fitted in log space to keep them positive, but the
    exponents are otherwise unconstrained -- without bounds, trf can
    push a or c large enough that (D/D_REF)^a overflows for D > D_REF
    (i.e. for most of the sample). That produces inf residuals mid-fit;
    the optimizer can still report success, but at a bad point, since
    the gradient off an inf is meaningless. Bounds keep the search in
    the physically sensible region.

    bounds : dict of key -> (lo, hi) in natural (not log) units, or
        None for the defaults below. Only keys present in fit_keys are
        used. Widen if a genuine optimum sits on an edge.
    """
    drop = set()
    if fix_c:
        drop.add('c')
    if not use_y0 and use_y0 is not None:
        drop.add('y0')
    elif use_y0 is None and par0.get('y0', 0.0) <= 0:
        drop.add('y0')
    fit_keys = [k for k in PARAM_KEYS if k not in drop]
    fixed = {'c': par0.get('c', 0.0)} if fix_c else {}
    if 'y0' in drop:
        fixed['y0'] = 0.0

    default_bounds = {
        'y0': (1e-8 * np.median(y), 1.0 * np.median(y)),
        'A':  (1e-4 * np.median(y), 1e2 * np.median(y)),
        'a':  (0.05, 4.0),
        'D0': (0.1 * np.min(d[d > 0]), 10.0 * np.max(d)),
        'B':  (1e-4 * np.median(y), 1e2 * np.median(y)),
        'b':  (0.05, 3.0),
        'c':  (0.0, 2.0),
    }
    if bounds:
        default_bounds.update(bounds)
    lo = np.log([max(default_bounds[k][0], 1e-30) for k in fit_keys])
    hi = np.log([default_bounds[k][1] for k in fit_keys])

    # A zero from NNLS enters as log(1e-30), which sits far below its
    # own lower bound -- clip the seed into range rather than let trf
    # start outside the box.
    seed_scale = {'y0': np.median(y) * 1e-2, 'A': np.median(y),
                  'a': 1.0, 'D0': np.median(d), 'B': np.median(y),
                  'b': 1.0, 'c': 0.3}
    x0 = np.log([par0[k] if par0.get(k, 0.0) > 0 else seed_scale[k]
                 for k in fit_keys])
    x0 = np.clip(x0, lo, hi)
    ly = np.log(y)

    def unpack(x):
        par = dict(zip(fit_keys, np.exp(x)))
        par.update(fixed)
        return par

    def resid(x):
        with np.errstate(over='ignore', invalid='ignore'):
            m = source_model(d, p, **unpack(x))
        return np.log(np.maximum(m, 1e-30)) - ly

    out = least_squares(resid, x0, bounds=(lo, hi), method='trf',
                        loss='soft_l1', max_nfev=200)
    par = unpack(out.x)

    if verbose:
        print("Log-space polish:")
        for key in PARAM_KEYS:
            tag = '  [fixed]' if key in fixed else ''
            print(f"  {key:3s} = {par[key]:.6g}   "
                  f"(NNLS: {par0.get(key, 0.0):.6g}){tag}")
        print(f"  cost = {out.cost:.6g},  success = {out.success}")
        for i, key in enumerate(fit_keys):
            if np.isclose(out.x[i], lo[i], atol=1e-6) or \
               np.isclose(out.x[i], hi[i], atol=1e-6):
                print(f"  WARNING: {key} sitting on its bound "
                      f"({default_bounds[key]}) -- widen if this looks wrong")

    return par, out


# =============================================================================
# SCORING
# =============================================================================

def score(d, p, y, par, med=None, dc=None, pc=None, cnt=None, label=''):
    """
    Report the correlations that matter.

    r_full : against raw targets.  Includes the irreducible within-cell
             scatter, so it cannot exceed the r_ceiling from
             cell_spread.  Directly comparable to the MLP benchmark.
    r_form : against the binned medians -- how well the functional form
             matches the surface, with the scatter removed.  Computed
             count-weighted when cnt is supplied, which is the fairer
             version: unweighted, a sparse corner cell counts as much
             as one holding 100k points.

    Also reports pred/obs quantiles.  r is blind to a constant
    multiplicative offset, and CAM needs the right magnitude, not just
    the right ranking.  If the 16-84% spread is symmetric in log space,
    it gives sigma for a stochastic implementation directly.
    """
    pred = source_model(d, p, **par)
    ok = np.isfinite(pred) & (pred > 0) & (y > 0)
    r_full = np.corrcoef(np.log(pred[ok]), np.log(y[ok]))[0, 1]

    ratio = pred[ok] / y[ok]
    q16, q50, q84 = np.percentile(ratio, [16, 50, 84])
    sigma = 0.5 * (np.log(q84) - np.log(q16))

    print(f"--- {label} ---")
    print(f"  r_full (log)     = {r_full:.4f}")
    print(f"  pred/obs median  = {q50:.3f}   16-84%: {q16:.3f}-{q84:.3f}")
    print(f"  implied sigma    = {sigma:.3f}  (log units, for stochastic)")

    r_form = None
    if med is not None:
        DD, PP = np.meshgrid(dc, pc, indexing='ij')
        pm = source_model(DD, PP, **par)
        m = np.isfinite(med) & np.isfinite(pm) & (pm > 0) & (med > 0)
        lpm, lmd = np.log(pm[m]), np.log(med[m])
        if cnt is not None:
            w = cnt[m].astype(float)
            w /= w.sum()
            mu_p, mu_m = np.sum(w * lpm), np.sum(w * lmd)
            cov = np.sum(w * (lpm - mu_p) * (lmd - mu_m))
            r_form = cov / np.sqrt(np.sum(w * (lpm - mu_p)**2) *
                                   np.sum(w * (lmd - mu_m)**2))
            tag = 'count-weighted'
        else:
            r_form = np.corrcoef(lpm, lmd)[0, 1]
            tag = 'unweighted'
        print(f"  r_form (surface) = {r_form:.4f}  [{tag}]")

    return dict(r_full=r_full, r_form=r_form, ratio_median=q50,
                ratio_q16=q16, ratio_q84=q84, sigma=sigma)


# =============================================================================
# RESIDUAL STRUCTURE
# =============================================================================

def residual_map(d, p, y, par, d_edges, p_edges, min_count=200,
                 verbose=True):
    """
    Median log(pred/obs) per (D, P) cell, on the same bins as
    bin_shape_1d.

    Structured residuals -- coherent blocks of one sign, or a monotone
    trend along either axis -- mean the form cannot reach the shape of
    the surface.  Unstructured, sign-alternating residuals mean the
    systematic signal has been extracted and what remains is noise.

    The printed diagnostics quantify this so the judgement does not
    rest on reading patterns into a colourmap.
    """
    pred = source_model(d, p, **par)
    ok = np.isfinite(pred) & (pred > 0) & (y > 0)
    lr = np.log(pred[ok]) - np.log(y[ok])

    di = np.clip(np.digitize(d[ok], d_edges) - 1, 0, len(d_edges) - 2)
    pi = np.clip(np.digitize(p[ok], p_edges) - 1, 0, len(p_edges) - 2)
    shape = (len(d_edges) - 1, len(p_edges) - 1)
    flat = di * shape[1] + pi

    order = np.argsort(flat, kind='stable')
    fs, lrs = flat[order], lr[order]
    starts = np.searchsorted(fs, np.arange(shape[0] * shape[1]))
    ends = np.searchsorted(fs, np.arange(shape[0] * shape[1]), side='right')
    cnt = (ends - starts).reshape(shape)

    res = np.full(shape, np.nan)
    for k in np.flatnonzero(cnt.ravel() >= min_count):
        res.ravel()[k] = np.median(lrs[starts[k]:ends[k]])

    if verbose:
        from scipy.stats import spearmanr
        good = np.isfinite(res)
        ii, jj = np.where(good)
        w = np.where(good, cnt, 0).astype(float)
        w /= w.sum()
        print(f"residual map: {good.sum()} of {good.size} cells")
        print(f"  spread (std over cells)      = {np.nanstd(res):.4f}")
        print(f"  max |median residual|        = {np.nanmax(np.abs(res)):.4f}")
        print(f"  count-weighted mean |res|    = "
              f"{np.sum(w[good] * np.abs(res[good])):.4f}")
        print(f"  fraction of points |res|>0.3 = "
              f"{np.sum(w[good] * (np.abs(res[good]) > 0.3)):.3f}")
        print(f"  Spearman vs D bin = {spearmanr(ii, res[good]).statistic:+.3f}")
        print(f"  Spearman vs P bin = {spearmanr(jj, res[good]).statistic:+.3f}")

    return res, cnt


def plot_residual_map(res, dc, pc, ax=None, vmax=None):
    """Diverging map of median log(pred/obs), centred on zero."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5))
    if vmax is None:
        vmax = np.nanmax(np.abs(res))
    im = ax.pcolormesh(np.log10(pc), np.log10(dc), res, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='nearest')
    ax.set(xlabel='log10 precl', ylabel='log10 D')
    plt.colorbar(im, ax=ax, label='median log(pred/obs)')
    return ax


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_models(d, p, y, d_edges, p_edges, med=None, dc=None, pc=None,
                   cnt=None, sample=100_000, **grid_kw):
    """
    Fit both the six-parameter (c = 0) and seven-parameter forms and
    report them side by side.

    The seven-parameter model will always fit the training hemisphere
    at least as well -- it contains the six-parameter model as a
    special case.  That is not evidence for it.  The decision belongs
    to the SH -> NH transfer test: run this on SH, then score both
    parameter sets on NH.  If the seven-parameter version transfers as
    well or better, the modulation is physical.  If it fits SH better
    and transfers worse, it has absorbed the frontal collinearity of
    the SH sample and the six-parameter form is the honest one.
    """
    out = {}
    for name, cg, fix in [('6-param', np.array([0.0]), True),
                          ('7-param', None, False)]:
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        g = fit_nnls_grid(d, p, y, c_grid=cg, sample=sample, **grid_kw)
        par, _ = fit_log_polish(d, p, y, g['params'], fix_c=fix)
        s = score(d, p, y, par, med=med, dc=dc, pc=pc, cnt=cnt, label=name)
        res, rcnt = residual_map(d, p, y, par, d_edges, p_edges)
        out[name] = dict(grid=g, params=par, score=s, res=res)

    return out