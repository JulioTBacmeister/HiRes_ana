"""
mlp_to_pptx.py
==============
Export MLP training results to a PowerPoint deck.

Slides
------
  1 — Run config + distribution diagnostics (SH test)
  2 — Predicted vs actual + feature importance (SH test)
  3 — Transfer diagnostics (NH, optional)

Typical notebook usage
----------------------
    import mlp_to_pptx
    importlib.reload(mlp_to_pptx)

    mlp_to_pptx.export_mlp_pptx(
        mlp_results      = mlp_results,
        hyperparameters  = MyHyperparameters,
        case_name        = case_name,
        pptx_out         = f"/path/to/{case_name}.pptx",
        transfer_results = {
            'y_pred' : y_pred_2,
            'y_targ' : yv_2,
            'label'  : 'NH transfer (El2)',
        },
    )

The JS assembler (make_mlp_slides.js) must live in the same directory.
Intermediate PNGs go to a temp directory and are cleaned up automatically
unless keep_pngs=True.
"""

import os
import json
import subprocess
import tempfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_mlp_pptx(
    mlp_results,
    hyperparameters,
    case_name,
    pptx_out,
    transfer_results = None,   # dict with keys y_pred, y_targ, label  (optional)
    top_n            = 20,
    keep_pngs        = False,
    js_script        = None,
    verbose          = True,
):
    """
    Generate a 2- or 3-slide PowerPoint from an mlp_utils results dict.

    Parameters
    ----------
    mlp_results      : dict returned by fit_mlp_general
    hyperparameters  : dict returned by Predi.reset_hyperparameters()
    case_name        : str   e.g. "c124_dyamond1_prod2"
    pptx_out         : str   full path for the output .pptx file
    transfer_results : dict or None
        If provided, a 3rd slide is added with transfer diagnostics.
        Keys:
            'y_pred'  np.ndarray  predictions on new domain (physical units)
            'y_targ'  np.ndarray  targets on new domain (physical units)
            'label'   str         e.g. "NH transfer (El2)"
    top_n            : int   features shown in importance bar chart
    keep_pngs        : bool  copy PNGs alongside PPTX (default False)
    js_script        : str or None  path to make_mlp_slides.js
    verbose          : bool
    """
    if js_script is None:
        js_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "make_mlp_slides.js")
    if not os.path.isfile(js_script):
        raise FileNotFoundError(
            f"make_mlp_slides.js not found at: {js_script}\n"
            f"Put it in the same directory as mlp_to_pptx.py."
        )

    work_dir = tempfile.mkdtemp(prefix="mlp_pptx_")
    try:
        fig1_path = os.path.join(work_dir, "fig_distributions.png")
        fig2_path = os.path.join(work_dir, "fig_results.png")
        fig3_path = os.path.join(work_dir, "fig_transfer.png") \
                    if transfer_results is not None else None

        if verbose:
            print("mlp_to_pptx: generating figures …")
        _make_distribution_figure(mlp_results, fig1_path)
        _make_results_figure(mlp_results, fig2_path, top_n=top_n)
        if transfer_results is not None:
            _make_transfer_figure(transfer_results, fig3_path)

        meta = _build_meta(
            mlp_results, hyperparameters, case_name,
            fig1_path, fig2_path, fig3_path, pptx_out,
            transfer_results=transfer_results,
        )

        meta_path = os.path.join(work_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if verbose:
            print("mlp_to_pptx: assembling slides …")
        result = subprocess.run(
            ["node", js_script, meta_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"make_mlp_slides.js failed:\n{result.stderr}")
        if verbose:
            print(result.stdout.strip())
            print(f"mlp_to_pptx: done  →  {pptx_out}")

        if keep_pngs:
            stem = os.path.splitext(pptx_out)[0]
            shutil.copy(fig1_path, stem + "_fig1_distributions.png")
            shutil.copy(fig2_path, stem + "_fig2_results.png")
            if fig3_path:
                shutil.copy(fig3_path, stem + "_fig3_transfer.png")
            if verbose:
                print("  PNGs saved alongside PPTX (keep_pngs=True)")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def _build_meta(results, hp, case_name,
                fig1_path, fig2_path, fig3_path, pptx_out,
                transfer_results=None):
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
        "fig1":        fig1_path,
        "fig2":        fig2_path,
        "fig3":        fig3_path,        # None if no transfer
        "pptx_out":    pptx_out,
    }

    # transfer summary stats (shown on slide 3 header cards)
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
# Figure 1 — Distribution diagnostics (4-panel, SH test)
# ---------------------------------------------------------------------------

def _make_distribution_figure(results, out_path):
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
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Predicted vs actual + feature importance
# ---------------------------------------------------------------------------

def _make_results_figure(results, out_path, top_n=20):
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
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Transfer diagnostics (2-panel: log scatter + residuals)
# ---------------------------------------------------------------------------

def _make_transfer_figure(transfer_results, out_path):
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
    ax.scatter(ly_targ, ly_pred, alpha=0.3, s=8, color='#4472C4')
    lims = [min(ly_targ.min(), ly_pred.min()), max(ly_targ.max(), ly_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.2)
    ax.set_xlabel('log(actual)', fontsize=11)
    ax.set_ylabel('log(predicted)', fontsize=11)
    ax.set_title(f'Log-space scatter  r={r_log:.3f}', fontsize=12)

    # panel 2 — residuals
    ax = axes[1]
    ax.hist(resid, bins=60, color='#4472C4', density=True)
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
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
