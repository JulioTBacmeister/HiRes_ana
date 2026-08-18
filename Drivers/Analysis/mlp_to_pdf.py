"""
mlp_to_pdf.py
=============
Export MLP training results to a multi-page PDF.

Pure-Python replacement for the mlp_to_pptx.py + make_mlp_slides.js
pipeline: figures go straight into the PDF via matplotlib's PdfPages,
so there is no Node step, no temp PNGs, no meta.json.

Pages (single run)
------------------
  1 — Run config summary (text page)
  2 — Distribution diagnostics (SH test)
  3 — Predicted vs actual + feature importance (SH test)
  4 — Transfer diagnostics (NH, optional)

Typical notebook usage
----------------------
    import mlp_to_pdf
    importlib.reload(mlp_to_pdf)

    mlp_to_pdf.export_mlp_pdf(
        mlp_results      = mlp_results,
        hyperparameters  = MyHyperparameters,
        case_name        = case_name,
        pdf_out          = f"/path/to/{case_name}.pdf",
        transfer_results = {
            'y_pred' : y_pred_2,
            'y_targ' : yv_2,
            'label'  : 'NH transfer (El2)',
        },
    )

Sweep usage
-----------
    mlp_to_pdf.export_sweep_pdf(
        sweep_results = sweep_results,   # list of (run_name, mlp_results,
        case_name     = case_name,       #          transfer_results, hp)
        pdf_out       = f"/path/to/{case_name}_sweep.pdf",
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# Public API — single run
# ---------------------------------------------------------------------------

def export_mlp_pdf(
    mlp_results,
    hyperparameters,
    case_name,
    pdf_out,
    transfer_results = None,   # dict with keys y_pred, y_targ, label  (optional)
    top_n            = 20,
    verbose          = True,
):
    """
    Generate a 3- or 4-page PDF from an mlp_utils results dict.

    Parameters
    ----------
    mlp_results      : dict returned by fit_mlp_general
    hyperparameters  : dict returned by Predi.reset_hyperparameters()
    case_name        : str   e.g. "c124_dyamond1_prod2"
    pdf_out          : str   full path for the output .pdf file
    transfer_results : dict or None
        If provided, a 4th page is added with transfer diagnostics.
        Keys:
            'y_pred'  np.ndarray  predictions on new domain (physical units)
            'y_targ'  np.ndarray  targets on new domain (physical units)
            'label'   str         e.g. "NH transfer (El2)"
    top_n            : int   features shown in importance bar chart
    verbose          : bool
    """
    meta = _build_meta(mlp_results, hyperparameters, case_name,
                       transfer_results=transfer_results)

    if verbose:
        print("mlp_to_pdf: generating pages …")

    with PdfPages(pdf_out) as pdf:
        _save(pdf, _make_config_page(meta))
        _save(pdf, _make_distribution_figure(mlp_results))
        _save(pdf, _make_results_figure(mlp_results, top_n=top_n))
        if transfer_results is not None:
            _save(pdf, _make_transfer_figure(transfer_results))

    if verbose:
        print(f"mlp_to_pdf: done  →  {pdf_out}")


# ---------------------------------------------------------------------------
# Public API — sweep export
# ---------------------------------------------------------------------------

def export_sweep_pdf(
    sweep_results,
    case_name,
    pdf_out,
    top_n   = 20,
    verbose = True,
):
    """
    Generate a sweep summary PDF from a list of per-run results.

    Parameters
    ----------
    sweep_results : list of tuples
        Each tuple: (run_name, mlp_results, transfer_results, hyperparameters)
        where transfer_results is a dict with keys y_pred, y_targ, label
        (or None if no transfer was run).
    case_name : str
    pdf_out   : str   full path for the output .pdf file
    top_n     : int   features shown in per-run importance chart
    verbose   : bool
    """
    with PdfPages(pdf_out) as pdf:
        if verbose:
            print("mlp_to_pdf: generating sweep summary page …")
        _save(pdf, _make_summary_figure(sweep_results))

        for i, entry in enumerate(sweep_results):
            run_name, mlp_results, transfer_results, hp = entry
            if verbose:
                print(f"  [{i+1}/{len(sweep_results)}] {run_name} …")

            meta = _build_meta(mlp_results, hp, f"{case_name} — {run_name}",
                               transfer_results=transfer_results)
            _save(pdf, _make_config_page(meta))
            _save(pdf, _make_distribution_figure(mlp_results))
            _save(pdf, _make_results_figure(mlp_results, top_n=top_n))
            if transfer_results is not None:
                _save(pdf, _make_transfer_figure(transfer_results))

    if verbose:
        print(f"mlp_to_pdf: done  →  {pdf_out}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(pdf, fig):
    pdf.savefig(fig, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def _build_meta(results, hp, case_name, transfer_results=None):
    use_pred   = hp.get('use_predictors', [])
    xrange     = hp.get('xrange', None)
    yrange     = hp.get('yrange', None)
    z_targ_m   = hp.get('z_targ', None)

    lat_lon = (
        f"{xrange} x {yrange}" if (xrange and yrange)
        else f"full domain x {yrange}" if yrange
        else "see case"
    )
    z_targ_str = f"{z_targ_m/1000:.0f} km" if z_targ_m is not None else "?"

    meta = {
        "case":        case_name,
        "fields":      ", ".join(use_pred) if use_pred else "see case",
        "n_pred":      len(results['predictor_names']),
        "hidden_dims": str(hp.get('hidden_dims', '?')),
        "loss_power":  float(hp.get('loss_power', 2.0)),
        "lat_lon":     lat_lon,
        "z_targ":      z_targ_str,
        "train_n":     int(results['y_train'].shape[0]),
        "test_n":      int(results['y_test'].shape[0]),
        "train_r2":    float(results['r2_train']),
        "test_r2":     float(results['r2_test']),
        "test_r":      float(results['r_test']),
    }

    # transfer summary stats (shown on config page)
    if transfer_results is not None:
        eps     = 1e-12
        y_pred  = transfer_results['y_pred']
        y_targ  = transfer_results['y_targ']
        ly_pred = np.log(np.maximum(y_pred, eps))
        ly_targ = np.log(np.maximum(y_targ, eps))
        r_tr, _ = stats.pearsonr(ly_targ, ly_pred)
        r2_tr   = r2_score(ly_targ, ly_pred)
        meta["transfer_label"] = transfer_results.get('label', 'Transfer')
        meta["transfer_r"]     = float(r_tr)
        meta["transfer_r2"]    = float(r2_tr)
        meta["transfer_n"]     = int(len(y_targ))

    return meta


# ---------------------------------------------------------------------------
# Page 1 — Run config summary (text page)
# ---------------------------------------------------------------------------

def _make_config_page(meta):
    """Text-only page replicating the slide-1 config header."""
    rows = [
        ("Case",             meta['case']),
        ("Predictor fields", meta['fields']),
        ("N predictors",     str(meta['n_pred'])),
        ("Hidden dims",      meta['hidden_dims']),
        ("Loss power",       f"{meta['loss_power']:.0f}"),
        ("Domain (lon x lat)", meta['lat_lon']),
        ("Target level",     meta['z_targ']),
        ("Train / test n",   f"{meta['train_n']:,} / {meta['test_n']:,}"),
        ("Train R²",         f"{meta['train_r2']:.3f}"),
        ("Test R²",          f"{meta['test_r2']:.3f}"),
        ("Test r",           f"{meta['test_r']:.3f}"),
    ]
    if 'transfer_label' in meta:
        rows += [
            (meta['transfer_label'],
             f"r={meta['transfer_r']:.3f}   "
             f"R²={meta['transfer_r2']:.3f}   "
             f"n={meta['transfer_n']:,}"),
        ]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    ax.axis('off')

    ax.text(0.02, 0.97, meta['case'],
            fontsize=18, fontweight='bold', va='top',
            transform=ax.transAxes)
    ax.text(0.02, 0.91, 'MLP run configuration & scores',
            fontsize=12, color='#555555', va='top',
            transform=ax.transAxes)

    y = 0.82
    for label, value in rows:
        ax.text(0.04, y, label, fontsize=11, fontweight='bold',
                va='top', transform=ax.transAxes)
        ax.text(0.32, y, value, fontsize=11,
                va='top', transform=ax.transAxes, wrap=True)
        y -= 0.055

    return fig


# ---------------------------------------------------------------------------
# Page 2 — Distribution diagnostics (4-panel, SH test)
# ---------------------------------------------------------------------------

def _make_distribution_figure(results):
    y_test  = results['y_test']
    y_pred  = results['y_pred_test']
    y_train = results['y_train']

    eps      = 1e-12
    ly_test  = np.log(np.maximum(y_test,  eps))
    ly_pred  = np.log(np.maximum(y_pred,  eps))
    ly_train = np.log(np.maximum(y_train, eps))
    resid    = ly_pred - ly_test
    r2_log   = r2_score(ly_test, ly_pred)
    r_log, _ = stats.pearsonr(ly_test, ly_pred)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('white')

    # panel 1 — physical space
    ax   = axes[0]
    bins = np.linspace(0, np.percentile(np.concatenate([y_test, y_pred]), 99), 60)
    ax.hist(y_train, bins=bins, alpha=0.30, label='actual (train)',   density=True, color='grey')
    ax.hist(y_test,  bins=bins, alpha=0.55, label='actual (test)',    density=True, color='#4472C4')
    ax.hist(y_pred,  bins=bins, alpha=0.55, label='predicted (test)', density=True, color='#ED7D31')
    ax.set_xlabel('epwp  (physical)', fontsize=11)
    ax.set_ylabel('density', fontsize=11)
    ax.set_title('Physical-space distribution', fontsize=12)
    ax.legend(fontsize=9)

    # panel 2 — log space
    ax       = axes[1]
    all_log  = np.concatenate([ly_test, ly_pred, ly_train])
    bins_log = np.linspace(np.percentile(all_log, 1), np.percentile(all_log, 99), 60)
    ax.hist(ly_train, bins=bins_log, alpha=0.30, label='actual (train)',   density=True, color='grey')
    ax.hist(ly_test,  bins=bins_log, alpha=0.55, label='actual (test)',    density=True, color='#4472C4')
    ax.hist(ly_pred,  bins=bins_log, alpha=0.55, label='predicted (test)', density=True, color='#ED7D31')
    ax.set_xlabel('log(epwp)', fontsize=11)
    ax.set_ylabel('density', fontsize=11)
    ax.set_title('Log-space distribution', fontsize=12)
    ax.legend(fontsize=9)

    # panel 3 — log-log scatter
    ax   = axes[2]
    ax.scatter(ly_test, ly_pred, alpha=0.3, s=8, color='#4472C4')
    lims = [min(ly_test.min(), ly_pred.min()), max(ly_test.max(), ly_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.2)
    ax.set_xlabel('log(actual)', fontsize=11)
    ax.set_ylabel('log(predicted)', fontsize=11)
    ax.set_title(f'Log-space scatter  r={r_log:.3f}', fontsize=12)

    # panel 4 — residuals
    ax = axes[3]
    ax.hist(resid, bins=60, color='#4472C4', density=True)
    ax.axvline(0,            color='r', lw=1.5, linestyle='--')
    ax.axvline(resid.mean(), color='k', lw=1.0, linestyle=':',
               label=f'mean={resid.mean():.3f}')
    ax.set_xlabel('log(pred) − log(actual)', fontsize=11)
    ax.set_ylabel('density', fontsize=11)
    ax.set_title(f'Log residuals  std={resid.std():.3f}', fontsize=12)
    ax.legend(fontsize=9)

    fig.suptitle(
        f'Distribution diagnostics  |  '
        f'Test R²={results["r2_test"]:.3f}   '
        f'Log-space R²={r2_log:.3f}   '
        f'r={results["r_test"]:.3f}',
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page 3 — Predicted vs actual + feature importance
# ---------------------------------------------------------------------------

def _make_results_figure(results, top_n=20):
    y_test = results['y_test']
    y_pred = results['y_pred_test']
    names  = results['predictor_names']
    imp    = results['importances']
    order  = results['importance_order']
    top_n  = min(top_n, len(names))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.35, s=10, color='#4472C4')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.2)
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(
        f'MLP  Predicted vs Actual (test)\n'
        f'Train R²={results["r2_train"]:.3f}   '
        f'Test R²={results["r2_test"]:.3f}   '
        f'r={results["r_test"]:.3f}',
        fontsize=12,
    )

    ax   = axes[1]
    idx  = order[:top_n]
    vals = imp[idx][::-1]
    nms  = [names[i] for i in idx[::-1]]
    cols = ['#4472C4' if v >= 0 else '#C00000' for v in vals]
    ax.barh(range(top_n), vals, color=cols)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(nms, fontsize=8)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('Permutation importance (drop in test R²)', fontsize=11)
    ax.set_title(f'Top {top_n} features  (MLP)', fontsize=12)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sweep page 1 — Summary bar chart
# ---------------------------------------------------------------------------

def _make_summary_figure(sweep_results):
    """Grouped bar chart comparing Test R², Test r, Transfer r across all runs."""
    run_names  = [e[0] for e in sweep_results]
    test_r2    = [float(e[1]['r2_test'])  for e in sweep_results]
    test_r     = [float(e[1]['r_test'])   for e in sweep_results]
    train_r2   = [float(e[1]['r2_train']) for e in sweep_results]

    eps = 1e-12
    transfer_r = []
    for _, _, tr, _ in sweep_results:
        if tr is not None:
            ly_p = np.log(np.maximum(tr['y_pred'], eps))
            ly_t = np.log(np.maximum(tr['y_targ'], eps))
            r, _ = stats.pearsonr(ly_t, ly_p)
            transfer_r.append(float(r))
        else:
            transfer_r.append(None)

    n    = len(run_names)
    x    = np.arange(n)
    w    = 0.20
    has_transfer = any(r is not None for r in transfer_r)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.patch.set_facecolor('white')

    # --- left panel: R² bars ---
    ax = axes[0]
    ax.bar(x - w,   train_r2, w*1.8, label='Train R²', color='#9DC3E6', edgecolor='k', lw=0.5)
    ax.bar(x + 0.0, test_r2,  w*1.8, label='Test R²',  color='#4472C4', edgecolor='k', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=35, ha='right', fontsize=10)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=10)
    ax.set_title('Train R²  vs  Test R²', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    # value labels
    for xi, (tr2, te2) in enumerate(zip(train_r2, test_r2)):
        ax.text(xi - w, tr2 + 0.01, f'{tr2:.3f}', ha='center', va='bottom', fontsize=7.5, color='#4472C4')
        ax.text(xi,     te2 + 0.01, f'{te2:.3f}', ha='center', va='bottom', fontsize=7.5, color='#1F3864')

    # --- right panel: r bars ---
    ax = axes[1]
    ax.bar(x - w/2, test_r, w*1.8, label='Test r (SH)',      color='#4472C4', edgecolor='k', lw=0.5)
    if has_transfer:
        tr_vals = [r if r is not None else 0 for r in transfer_r]
        ax.bar(x + w/2, tr_vals, w*1.8, label='Transfer r (NH)', color='#70AD47', edgecolor='k', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=35, ha='right', fontsize=10)
    ax.set_ylabel('Pearson r', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=10)
    ax.set_title('Test r (SH)  vs  Transfer r (NH)', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    for xi, (tr, trf) in enumerate(zip(test_r, transfer_r)):
        ax.text(xi - w/2, tr  + 0.01, f'{tr:.3f}',  ha='center', va='bottom', fontsize=7.5, color='#1F3864')
        if trf is not None:
            ax.text(xi + w/2, trf + 0.01, f'{trf:.3f}', ha='center', va='bottom', fontsize=7.5, color='#375623')

    fig.suptitle('Predictor Sweep Summary', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page 4 — Transfer diagnostics (2-panel: log scatter + residuals)
# ---------------------------------------------------------------------------

def _make_transfer_figure(transfer_results):
    y_pred = transfer_results['y_pred']
    y_targ = transfer_results['y_targ']
    label  = transfer_results.get('label', 'Transfer')

    eps     = 1e-12
    ly_pred = np.log(np.maximum(y_pred, eps))
    ly_targ = np.log(np.maximum(y_targ, eps))
    resid   = ly_pred - ly_targ
    r_log, _ = stats.pearsonr(ly_targ, ly_pred)
    r2_log   = r2_score(ly_targ, ly_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')

    # panel 1 — log-log scatter
    ax   = axes[0]
    ax.scatter(ly_targ, ly_pred, alpha=0.3, s=8, color='#70AD47')
    lims = [min(ly_targ.min(), ly_pred.min()), max(ly_targ.max(), ly_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.2)
    ax.set_xlabel('log(actual)', fontsize=11)
    ax.set_ylabel('log(predicted)', fontsize=11)
    ax.set_title(f'Log-space scatter  r={r_log:.3f}', fontsize=12)

    # panel 2 — residuals
    ax = axes[1]
    ax.hist(resid, bins=60, color='#70AD47', density=True)
    ax.axvline(0,            color='r', lw=1.5, linestyle='--')
    ax.axvline(resid.mean(), color='k', lw=1.0, linestyle=':',
               label=f'mean={resid.mean():.3f}')
    ax.set_xlabel('log(pred) − log(actual)', fontsize=11)
    ax.set_ylabel('density', fontsize=11)
    ax.set_title(f'Log residuals  std={resid.std():.3f}', fontsize=12)
    ax.legend(fontsize=9)

    fig.suptitle(
        f'{label}  |  '
        f'R²={r2_log:.3f}   r={r_log:.3f}   n={len(y_targ):,}',
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    return fig