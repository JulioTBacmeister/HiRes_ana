"""
MLP regressor for gravity-wave momentum-flux prediction.

Designed as a drop-in companion to fit_rf_general / plot_rf_results in
random_forest.py.  The results dict has identical keys so plot_rf_results
works unchanged.

Splitting strategy
------------------
Random splits are intentionally NOT used because temporal autocorrelation
in atmospheric model output would inflate test scores.  All splits are
chronological:

  - If train_interval / test_interval are given: use those time masks.
  - Otherwise: sort events by event_times (if provided) or by array order,
    then use the first (1-test_size) fraction for training and the last
    test_size fraction for testing.
  - The early-stopping validation set is always the last 10 % of the
    training block (never a random subsample).

Target normalisation
--------------------
epwp values are ~0.001–0.05.  log1p(x) ≈ x for x << 1, so it provides
no benefit.  Instead we apply two steps:
  1. np.log(y)          — spreads the log-normal distribution (values go
                          from ~[-7, -3.5] with std ≈ 0.9)
  2. StandardScaler     — zero-mean / unit-variance so the loss is
                          scale-invariant

Back-transform: inverse-scale → np.exp().

Regularization stack:
  - BatchNorm1d  (stabilises training, implicit regularisation)
  - Dropout      (default 0.2)
  - AdamW L2 weight decay (default 1e-5)
  - ReduceLROnPlateau scheduler
  - Early stopping on held-out validation loss (last 10 % of train)
"""

import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Feed-forward MLP with BatchNorm + LeakyReLU + Dropout after every layer.

    Parameters
    ----------
    input_dim   : int
    hidden_dims : tuple of int   default (256, 256, 128)
    dropout     : float          default 0.2
    """

    def __init__(self, input_dim, hidden_dims=(256, 256, 128), dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def permutation_importance_mlp(model, X_test_t, y_test_orig,
                                target_scaler, device, n_repeats=5):
    """
    Model-agnostic permutation importance: drop in R² when feature i is shuffled.

    Parameters
    ----------
    model         : MLP in eval mode
    X_test_t      : torch.FloatTensor, scaled test features
    y_test_orig   : np.ndarray, test targets in physical space
    target_scaler : fitted StandardScaler on log(y_train) — used to invert
    device        : str
    n_repeats     : int — permutations per feature (results are averaged)

    Returns
    -------
    importances : np.ndarray, shape (n_features,)
        Positive = important.  Negative = feature hurts generalisation.
    """
    model.eval()
    rng = np.random.default_rng(42)

    def _predict(X_t):
        with torch.no_grad():
            out = model(X_t.to(device)).cpu().numpy()
        log_y = target_scaler.inverse_transform(out.reshape(-1, 1)).flatten()
        return np.exp(log_y)

    base_r2 = r2_score(y_test_orig, _predict(X_test_t))
    n_features = X_test_t.shape[1]
    importances = np.zeros(n_features)

    for fi in range(n_features):
        drop = 0.0
        for _ in range(n_repeats):
            X_perm = X_test_t.clone()
            X_perm[:, fi] = X_perm[rng.permutation(X_perm.shape[0]), fi]
            drop += base_r2 - r2_score(y_test_orig, _predict(X_perm))
        importances[fi] = drop / n_repeats

    return importances


# ---------------------------------------------------------------------------
# Main fit function — mirrors fit_rf_general API exactly
# ---------------------------------------------------------------------------

def fit_mlp_general(
    predictors,
    predictor_names,
    target,
    event_times=None,
    train_interval=None,
    test_interval=None,
    test_size=0.2,
    # --- architecture ---
    hidden_dims=(256, 256, 128),
    dropout=0.2,
    # --- training ---
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    patience=40,
    max_epochs=500,
    # --- lr schedule ---
    lr_patience=10,
    lr_factor=0.5,
    min_lr=1e-6,
    # --- loss ---
    loss_power=2.0,   # 2 = MSE, 3 = cubic (penalises large errors more)
    # --- predictor transforms ---
    log_predictor_patterns=None,  # e.g. ['tilt'] — name substrings to log-transform
    log_predictor_eps=1e-30,      # added before log to guard against zeros
    # --- misc ---
    random_state=42,
    device=None,
    verbose=True,
):
    """
    Fit a regularised MLP regressor.  API mirrors fit_rf_general.

    All splits are chronological — random splits are never used.
    Target is normalised as StandardScaler(log(y)) before training.

    Parameters
    ----------
    predictors      : list of np.ndarray, each shape (n_events,)
    predictor_names : list of str
    target          : np.ndarray (n_events,)  — epwp in physical units, > 0
    event_times     : np.ndarray or None
    train_interval  : (t_start, t_end) or None
    test_interval   : (t_start, t_end) or None
    test_size       : float — fraction for test when no interval is given
    hidden_dims     : tuple of int
    dropout         : float
    batch_size      : int
    lr              : float
    weight_decay    : float
    patience        : int — early-stop patience (epochs)
    max_epochs      : int
    lr_patience            : int — epochs before ReduceLROnPlateau fires
    lr_factor              : float — LR reduction factor
    min_lr                 : float — LR floor
    loss_power             : float — exponent of the error term in the loss.
        2.0 = MSE (default).  3.0 = cubic: penalises large errors ~9× more
        than errors half their size, focusing training on the biggest events.
        Gradient of |e|^p is p·e·|e|^(p-2), smooth and well-defined at 0.
    log_predictor_patterns : list of str or None
        Name substrings (case-insensitive) identifying predictors to
        log-transform before StandardScaler.  e.g. ['tilt'] transforms
        every predictor whose name contains 'tilt'.  Applied as
        log(|x| + eps).  Use for right-skewed, non-negative quantities.
    log_predictor_eps      : float — epsilon added inside log (default 1e-30)
    random_state           : int
    device                 : str or None
    verbose                : bool

    Returns
    -------
    model   : MLP in eval mode
    results : dict  (same keys as fit_rf_general — compatible with plot_rf_results)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(random_state)

    # ------------------------------------------------------------------ data
    assert len(predictors) == len(predictor_names)
    X = np.column_stack(predictors).astype(np.float32)
    y = target.astype(np.float32)

    # log-transform the target (requires y > 0; epwp is always positive)
    assert np.all(y > 0), "target must be strictly positive (got zeros or negatives)"
    log_y = np.log(y)

    if verbose:
        print(f"Feature matrix : {X.shape[0]} events x {X.shape[1]} predictors")
        print(f"Target (raw)   : mean={y.mean():.4f}  std={y.std():.4f}")
        print(f"Target (log)   : mean={log_y.mean():.3f}  std={log_y.std():.3f}"
              f"  device={device}")
        print(f"Architecture   : {hidden_dims} ")
        print(f"lr_patience    : {lr_patience} ")
        print(f"minimum lr     : {min_lr} ")
        

    # ------------------------------------------------------- chronological split
    if train_interval is not None:
        assert event_times is not None, "event_times required when train_interval is set"
        t_lo, t_hi = train_interval
        train_mask = (event_times >= t_lo) & (event_times < t_hi)
        if test_interval is not None:
            v_lo, v_hi = test_interval
            test_mask = (event_times >= v_lo) & (event_times < v_hi)
        else:
            test_mask = ~train_mask
        X_train_raw, logy_train, y_train_orig = (
            X[train_mask], log_y[train_mask], y[train_mask]
        )
        X_test_raw, logy_test, y_test_orig = (
            X[test_mask], log_y[test_mask], y[test_mask]
        )
        split_mode = 'time-interval'
        if verbose:
            print(f"Interval split : {train_mask.sum()} train, {test_mask.sum()} test")
    else:
        if event_times is not None:
            order = np.argsort(event_times, kind='stable')
        else:
            order = np.arange(len(y))
        n_train = int(len(order) * (1.0 - test_size))
        tr_idx_ = order[:n_train]
        te_idx_ = order[n_train:]
        X_train_raw, logy_train, y_train_orig = X[tr_idx_], log_y[tr_idx_], y[tr_idx_]
        X_test_raw,  logy_test,  y_test_orig  = X[te_idx_], log_y[te_idx_], y[te_idx_]
        split_mode = 'chronological'
        if verbose:
            print(f"Chronological split : {len(tr_idx_)} train  "
                  f"(first {100*(1-test_size):.0f}%),  "
                  f"{len(te_idx_)} test  (last {100*test_size:.0f}%)")

    # --------------------------------- per-predictor log transforms (optional)
    log_pred_mask = np.zeros(X.shape[1], dtype=bool)
    if log_predictor_patterns:
        for fi, name in enumerate(predictor_names):
            if any(pat.lower() in name.lower() for pat in log_predictor_patterns):
                log_pred_mask[fi] = True
        n_logged = log_pred_mask.sum()
        if verbose:
            print(f"Log-transforming {n_logged} predictor(s) matching "
                  f"{log_predictor_patterns}")

    def _transform_predictors(X_raw):
        Xt = X_raw.copy()
        if log_pred_mask.any():
            Xt[:, log_pred_mask] = np.log(
                np.abs(Xt[:, log_pred_mask]) + log_predictor_eps
            )
        return Xt

    X_train_tr = _transform_predictors(X_train_raw)
    X_test_tr  = _transform_predictors(X_test_raw)

    # ----------------------------------------- scale features and log-target
    feat_scaler   = StandardScaler()
    X_train_s = feat_scaler.fit_transform(X_train_tr).astype(np.float32)
    X_test_s  = feat_scaler.transform(X_test_tr).astype(np.float32)

    # fit target scaler on training log-y only
    target_scaler = StandardScaler()
    y_train_m = target_scaler.fit_transform(
        logy_train.reshape(-1, 1)
    ).flatten().astype(np.float32)
    # test log-y will be inverted after prediction; no transform needed here

    if verbose:
        print(f"log(y) train   : mean={logy_train.mean():.3f}  "
              f"std={logy_train.std():.3f}  "
              f"→ normalised to mean≈0  std≈1")

    # Carve the last 10 % of the training block as validation (chronological)
    n_val   = max(2, int(0.1 * len(X_train_s)))
    tr_idx  = np.arange(0, len(X_train_s) - n_val)
    val_idx = np.arange(len(X_train_s) - n_val, len(X_train_s))

    X_tr_t   = torch.from_numpy(X_train_s[tr_idx])
    y_tr_t   = torch.from_numpy(y_train_m[tr_idx])
    X_val_t  = torch.from_numpy(X_train_s[val_idx])
    y_val_t  = torch.from_numpy(y_train_m[val_idx])
    X_test_t = torch.from_numpy(X_test_s)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(tr_idx) > batch_size),
    )

    # ------------------------------------------------------------ model
    model = MLP(X.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor,
        patience=lr_patience, min_lr=min_lr,
    )

    def criterion(pred, target):
        if loss_power == 2.0:
            return nn.functional.mse_loss(pred, target)
        # |e|^p  — gradient is p·e·|e|^(p-2), smooth at 0
        return (pred - target).abs().pow(loss_power).mean()

    if verbose:
        print(f"Loss           : |error|^{loss_power:.1f}"
              + ("  (MSE)" if loss_power == 2.0 else "  (cubic)" if loss_power == 3.0 else ""))

    best_val   = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    no_impr    = 0
    prev_lr    = lr

    for epoch in range(max_epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(X_val_t.to(device)), y_val_t.to(device)
            ).item()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_impr    = 0
        else:
            no_impr += 1
            if no_impr >= patience:
                if verbose:
                    print(f"  Early stop epoch {epoch+1:3d}  "
                          f"best_val_loss={best_val:.4f}  lr={current_lr:.2e}")
                break

        if verbose and (epoch + 1) % 25 == 0:
            lr_tag = f"  ** lr → {current_lr:.2e}" if current_lr < prev_lr else ""
            print(f"  Epoch {epoch+1:3d}  val_loss={val_loss:.4f}  "
                  f"lr={current_lr:.2e}{lr_tag}")

        prev_lr = current_lr

    model.load_state_dict(best_state)
    model.eval()

    # ---------------------------------------------------------- predict
    def _infer(X_np):
        Xt = _transform_predictors(X_np)
        t  = torch.from_numpy(feat_scaler.transform(Xt).astype(np.float32)).to(device)
        with torch.no_grad():
            out = model(t).cpu().numpy()
        # clip to ±8 sigma before inverse-transform to prevent exp overflow
        out_clipped = np.clip(out, -8.0, 8.0)
        log_y_pred  = target_scaler.inverse_transform(
            out_clipped.reshape(-1, 1)
        ).flatten()
        return np.exp(log_y_pred)

    y_pred_train = _infer(X_train_raw)
    y_pred_test  = _infer(X_test_raw)

    r2_train  = r2_score(y_train_orig, y_pred_train)
    r2_test_  = r2_score(y_test_orig,  y_pred_test)
    r_test, p_test = stats.pearsonr(y_test_orig, y_pred_test)

    if verbose:
        print(f"Train R²:  {r2_train:.3f}")
        print(f"Test  R²:  {r2_test_:.3f}")
        print(f"Test  r:   {r_test:.3f}  (p={p_test:.2e})")
        print("Computing permutation importance …")

    importances = permutation_importance_mlp(
        model, X_test_t, y_test_orig, target_scaler, device, n_repeats=5,
    )
    importance_order = np.argsort(importances)[::-1]

    results = {
        'X_train':          X_train_raw,
        'X_test':           X_test_raw,
        'y_train':          y_train_orig,
        'y_test':           y_test_orig,
        'y_pred_train':     y_pred_train,
        'y_pred_test':      y_pred_test,
        'r2_train':         r2_train,
        'r2_test':          r2_test_,
        'r_test':           r_test,
        'p_test':           p_test,
        'cv_scores':        None,
        'importances':      importances,
        'importance_order': importance_order,
        'predictor_names':  predictor_names,
        'split_mode':           split_mode,
        'feat_scaler':          feat_scaler,
        'target_scaler':        target_scaler,
        'log_pred_mask':        log_pred_mask,
        'log_predictor_patterns': log_predictor_patterns,
    }

    return model, results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mlp_results(results, top_n=None, desc=None ):
    """
    Predicted-vs-actual scatter + permutation-importance bar chart.
    Compatible with results from fit_rf_general too (importances key present).
    """
    predictor_names = results['predictor_names']
    n_p   = len(predictor_names)
    top_n = min(top_n or n_p, n_p)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    ax.scatter(results['y_test'], results['y_pred_test'],
               alpha=0.4, s=15, color='steelblue')
    lims = [
        min(results['y_test'].min(), results['y_pred_test'].min()),
        max(results['y_test'].max(), results['y_pred_test'].max()),
    ]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(
        f'MLP  Predicted vs Actual (test)\n'
        f'Train R²={results["r2_train"]:.3f}   '
        f'Test R²={results["r2_test"]:.3f}   '
        f'r={results["r_test"]:.3f}'
    )

    ax = axes[1]
    idx   = results['importance_order'][:top_n]
    vals  = results['importances'][idx][::-1]
    nms   = [predictor_names[i] for i in idx[::-1]]
    colors = ['steelblue' if v >= 0 else 'tomato' for v in vals]
    ax.barh(range(top_n), vals, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(nms, fontsize=8)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('Permutation importance (drop in test R²)')
    ax.set_title(f'Top {top_n} features  (MLP)')

    if desc is not None:
            fig.text(0.5, -0.02, desc, ha='center', va='top', fontsize=9,
                     transform=fig.transFigure)

    plt.tight_layout()
    plt.show()


def plot_predictor_distributions(results, top_n_hist=8):
    """
    Diagnose the predictor distributions to check whether StandardScaler
    is sufficient or whether some features need further treatment.

    Panels
    ------
    Row 1  (all predictors, sorted):
      - Skewness bar chart
      - Excess kurtosis bar chart
      - Pearson r with log(target) bar chart
    Row 2  (top_n_hist most skewed predictors):
      - Raw histogram
      - Scaled histogram (after StandardScaler fit on training data)

    Parameters
    ----------
    results    : dict as returned by fit_mlp_general
    top_n_hist : int — how many extreme-skewness predictors to show as histograms
    """
    from scipy.stats import skew, kurtosis

    X_raw    = results['X_train']          # (n_train, n_feat), unscaled
    y_orig   = results['y_train']
    names    = results['predictor_names']
    scaler   = results['feat_scaler']
    n_feat   = X_raw.shape[1]
    log_mask = results.get('log_pred_mask', np.zeros(n_feat, dtype=bool))

    # apply the same per-predictor log transforms used during training
    X_tr = X_raw.copy()
    if log_mask.any():
        X_tr[:, log_mask] = np.log(np.abs(X_tr[:, log_mask]) + 1e-30)

    X_scaled = scaler.transform(X_tr).astype(np.float32)
    log_y    = np.log(np.maximum(y_orig, 1e-12))

    # --- summary stats per feature (computed on transformed values) ---
    skewness = np.array([skew(X_tr[:, i])     for i in range(n_feat)])
    kurt     = np.array([kurtosis(X_tr[:, i]) for i in range(n_feat)])  # excess
    corr_logy= np.array([
        stats.pearsonr(X_tr[:, i], log_y)[0] for i in range(n_feat)
    ])

    sk_order   = np.argsort(np.abs(skewness))[::-1]
    kurt_order = np.argsort(np.abs(kurt))[::-1]
    corr_order = np.argsort(np.abs(corr_logy))[::-1]

    fig = plt.figure(figsize=(20, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    x_ticks = np.arange(n_feat)

    # --- row 0, col 0: skewness ---
    ax = fig.add_subplot(gs[0, 0])
    colors = ['tomato' if abs(s) > 2 else 'steelblue' for s in skewness[sk_order]]
    ax.bar(x_ticks, skewness[sk_order], color=colors, width=1.0)
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline( 2, color='r', lw=0.8, linestyle='--', label='|skew|=2')
    ax.axhline(-2, color='r', lw=0.8, linestyle='--')
    ax.set_xlabel('Predictor (sorted by |skewness|)')
    ax.set_ylabel('Skewness')
    ax.set_title('Predictor skewness\n(red = |skew| > 2, may need log-transform)')
    ax.legend(fontsize=8)

    # --- row 0, col 1: excess kurtosis ---
    ax = fig.add_subplot(gs[0, 1])
    colors = ['tomato' if abs(k) > 7 else 'steelblue' for k in kurt[kurt_order]]
    ax.bar(x_ticks, kurt[kurt_order], color=colors, width=1.0)
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline( 7, color='r', lw=0.8, linestyle='--', label='|kurt|=7')
    ax.axhline(-7, color='r', lw=0.8, linestyle='--')
    ax.set_xlabel('Predictor (sorted by |kurtosis|)')
    ax.set_ylabel('Excess kurtosis')
    ax.set_title('Predictor kurtosis\n(red = heavy tails, outliers likely)')
    ax.legend(fontsize=8)

    # --- row 0, col 2: correlation with log(target) ---
    ax = fig.add_subplot(gs[0, 2])
    colors = ['tomato' if abs(c) > 0.3 else 'steelblue' for c in corr_logy[corr_order]]
    ax.bar(x_ticks, corr_logy[corr_order], color=colors, width=1.0)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Predictor (sorted by |r|)')
    ax.set_ylabel('Pearson r with log(target)')
    ax.set_title('Predictor–target correlation\n(red = |r| > 0.3)')
    # label top 5
    for rank in range(min(5, n_feat)):
        fi = corr_order[rank]
        ax.text(rank, corr_logy[fi] + 0.01 * np.sign(corr_logy[fi]),
                names[fi].split('|')[0], fontsize=6, ha='center', rotation=90)

    # --- row 1: histograms of top_n_hist most-skewed predictors ---
    top_skew_idx = sk_order[:top_n_hist]
    gs2 = gs[1, :].subgridspec(2, top_n_hist, hspace=0.05, wspace=0.3)

    for col, fi in enumerate(top_skew_idx):
        raw    = X_tr[:, fi]     # transformed (log applied if in mask)
        raw_orig = X_raw[:, fi]  # always original for reference
        scaled = X_scaled[:, fi]
        was_logged = log_mask[fi]
        label  = names[fi]
        log_tag = '  [log-transformed]' if was_logged else ''

        # transformed (or raw if no log applied)
        ax_r = fig.add_subplot(gs2[0, col])
        color_r = 'darkorange' if was_logged else 'steelblue'
        ax_r.hist(raw, bins=40, color=color_r, density=True)
        ax_r.set_title(f'{label}{log_tag}\nskew={skewness[fi]:.1f}', fontsize=6.5)
        ax_r.set_ylabel('density' if col == 0 else '')
        ax_r.tick_params(labelbottom=False, labelsize=7)
        if col == 0:
            ax_r.set_ylabel('raw\ndensity', fontsize=8)

        # scaled
        ax_s = fig.add_subplot(gs2[1, col])
        ax_s.hist(scaled, bins=40, color='darkorange', density=True)
        ax_s.tick_params(labelsize=7)
        if col == 0:
            ax_s.set_ylabel('scaled\ndensity', fontsize=8)

    n_bad_skew = int((np.abs(skewness) > 2).sum())
    n_bad_kurt = int((np.abs(kurt) > 7).sum())
    n_logged   = int(log_mask.sum())
    log_note   = (f'  |  {n_logged} predictor(s) log-transformed '
                  f'[shown in orange]') if n_logged else ''
    fig.suptitle(
        f'{n_feat} predictors  |  {n_bad_skew} with |skew|>2  |  '
        f'{n_bad_kurt} with |kurt|>7{log_note}\n'
        f'(stats computed after transforms; bottom row = top-{top_n_hist} most skewed)',
        fontsize=10
    )
    plt.show()


def plot_distributions(results, desc=None, log_scale=True):
    """
    Four-panel diagnostic showing prediction quality and distribution match.

    Panels
    ------
    1. Physical space: PDF of actual vs predicted (test set)
    2. Log space:      PDF of log(actual) vs log(predicted)
    3. Scatter:        predicted vs actual in log space
    4. Residuals:      log(pred) - log(actual) histogram
    """
    y_test  = results['y_test']
    y_pred  = results['y_pred_test']
    y_train = results['y_train']
    y_pred_train = results['y_pred_train']

    # guard against zeros/negatives in predictions
    eps = 1e-12
    ly_test       = np.log(np.maximum(y_test,  eps))
    ly_pred_test  = np.log(np.maximum(y_pred,  eps))
    ly_train      = np.log(np.maximum(y_train, eps))
    ly_pred_train = np.log(np.maximum(y_pred_train, eps))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # --- panel 1: physical space PDFs ---
    ax = axes[0]
    bins = np.linspace(0, np.percentile(np.concatenate([y_test, y_pred]), 99), 60)
    ax.hist(y_test,  bins=bins, alpha=0.5, label='actual (test)',     density=True)
    ax.hist(y_pred,  bins=bins, alpha=0.5, label='predicted (test)',  density=True)
    ax.hist(y_train, bins=bins, alpha=0.3, label='actual (train)',    density=True, color='grey')
    ax.set_xlabel('epwp  (physical)')
    ax.set_ylabel('density')
    ax.set_title('Physical-space distribution')
    ax.legend(fontsize=8)

    # --- panel 2: log-space PDFs ---
    ax = axes[1]
    all_log = np.concatenate([ly_test, ly_pred_test, ly_train])
    bins_log = np.linspace(np.percentile(all_log, 1), np.percentile(all_log, 99), 60)
    ax.hist(ly_test,       bins=bins_log, alpha=0.5, label='actual (test)',    density=True)
    ax.hist(ly_pred_test,  bins=bins_log, alpha=0.5, label='predicted (test)', density=True)
    ax.hist(ly_train,      bins=bins_log, alpha=0.3, label='actual (train)',   density=True, color='grey')
    ax.set_xlabel('log(epwp)')
    ax.set_ylabel('density')
    ax.set_title('Log-space distribution')
    ax.legend(fontsize=8)

    # --- panel 3: log-log scatter ---
    ax = axes[2]
    ax.scatter(ly_test, ly_pred_test, alpha=0.3, s=10, color='steelblue')
    lims = [min(ly_test.min(), ly_pred_test.min()),
            max(ly_test.max(), ly_pred_test.max())]
    ax.plot(lims, lims, 'r--', lw=1)
    r_log, _ = stats.pearsonr(ly_test, ly_pred_test)
    ax.set_xlabel('log(actual)')
    ax.set_ylabel('log(predicted)')
    ax.set_title(f'Log-space scatter  r={r_log:.3f}')

    # --- panel 4: residuals ---
    ax = axes[3]
    resid = ly_pred_test - ly_test
    ax.hist(resid, bins=60, color='steelblue', density=True)
    ax.axvline(0, color='r', lw=1.5, linestyle='--')
    ax.axvline(resid.mean(), color='k', lw=1, linestyle=':',
               label=f'mean={resid.mean():.3f}')
    ax.set_xlabel('log(pred) − log(actual)')
    ax.set_ylabel('density')
    ax.set_title(f'Log residuals  std={resid.std():.3f}')
    ax.legend(fontsize=8)

    r2_log = r2_score(ly_test, ly_pred_test)
    fig.suptitle(
        f'Distribution diagnostics  |  '
        f'Test R²={results["r2_test"]:.3f}  '
        f'Log-space R²={r2_log:.3f}  '
        f'r={results["r_test"]:.3f}',
        fontsize=11, y=1.01
    )

    if desc is not None:
            fig.text(0.5, -0.02, desc, ha='center', va='top', fontsize=9,
                     transform=fig.transFigure)

    plt.tight_layout()
    plt.show()

################################################################
def save_mlp(model, results, path_stem,
             hidden_dims=(256, 256, 128), dropout=0.2,
             # training hyperparameters — only needed for reproducibility
             loss_power=2.0, log_predictor_eps=1e-30,
             lr=1e-3, weight_decay=1e-5, batch_size=256,
             random_state=42):
    """
    Save model weights and all objects needed for inference or retraining.

    Saves two files:
      <path_stem>.pt   — model weights (torch.save)
      <path_stem>.pkl  — scalers, architecture, provenance metadata

    Parameters
    ----------
    model       : trained MLP returned by fit_mlp_general
    results     : dict returned by fit_mlp_general
    path_stem   : str — e.g. '/path/to/gw_mlp_dyamond'
    hidden_dims : tuple — must match architecture used at training time
    dropout     : float — must match
    loss_power, log_predictor_eps, lr, weight_decay, batch_size,
    random_state : training hyperparameters for record-keeping / retraining
    """
    torch.save(model.state_dict(), f'{path_stem}.pt')

    y_train  = results['y_train']
    logy_train = np.log(y_train)

    meta = {
        # --- preprocessing (required for inference) ---
        'feat_scaler':            results['feat_scaler'],
        'target_scaler':          results['target_scaler'],
        'log_pred_mask':          results['log_pred_mask'],
        'log_predictor_patterns': results['log_predictor_patterns'],
        'predictor_names':        results['predictor_names'],

        # --- architecture (required to reconstruct MLP) ---
        'input_dim':              results['X_train'].shape[1],
        'hidden_dims':            hidden_dims,
        'dropout':                dropout,

        # --- training hyperparameters (reproducibility / retraining) ---
        'loss_power':             loss_power,
        'log_predictor_eps':      log_predictor_eps,
        'lr':                     lr,
        'weight_decay':           weight_decay,
        'batch_size':             batch_size,
        'random_state':           random_state,

        # --- training provenance (transfer diagnostics) ---
        'split_mode':             results['split_mode'],
        'train_y_mean':           float(y_train.mean()),
        'train_y_std':            float(y_train.std()),
        'train_logy_mean':        float(logy_train.mean()),
        'train_logy_std':         float(logy_train.std()),
        'train_r2':               float(results['r2_train']),
        'test_r2':                float(results['r2_test']),
        'test_r':                 float(results['r_test']),
    }

    with open(f'{path_stem}.pkl', 'wb') as f:
        pickle.dump(meta, f)

    print(f"Saved weights  → {path_stem}.pt")
    print(f"Saved metadata → {path_stem}.pkl")
    print(f"  Architecture : MLP({meta['input_dim']} → "
          f"{hidden_dims} → 1)  dropout={dropout}")
    print(f"  Predictors   : {len(meta['predictor_names'])} features")
    print(f"  Train scores : R²={meta['train_r2']:.3f}  "
          f"Test R²={meta['test_r2']:.3f}  r={meta['test_r']:.3f}")
    print(f"  Train y      : mean={meta['train_y_mean']:.4f}  "
          f"std={meta['train_y_std']:.4f}")
    print(f"  Train log(y) : mean={meta['train_logy_mean']:.3f}  "
          f"std={meta['train_logy_std']:.3f}")

#################################################################
def load_mlp(path_stem, device=None):
    """
    Reconstruct a trained MLP ready for inference.
    Architecture is read from the saved metadata — no need to pass hidden_dims.

    Returns
    -------
    model  : MLP in eval mode
    meta   : dict — contains everything saved by save_mlp
    device : str
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(f'{path_stem}.pkl', 'rb') as f:
        meta = pickle.load(f)

    model = MLP(meta['input_dim'],
                hidden_dims=meta['hidden_dims'],
                dropout=meta['dropout']).to(device)
    model.load_state_dict(
        torch.load(f'{path_stem}.pt', map_location=device)
    )
    model.eval()

    print(f"Loaded  ← {path_stem}.pt")
    print(f"  Architecture : MLP({meta['input_dim']} → "
          f"{meta['hidden_dims']} → 1)  dropout={meta['dropout']}")
    print(f"  Predictors   : {meta['predictor_names']}")
    print(f"  Train scores : R²={meta['train_r2']:.3f}  "
          f"Test R²={meta['test_r2']:.3f}  r={meta['test_r']:.3f}")
    print(f"  Train y      : mean={meta['train_y_mean']:.4f}  "
          f"std={meta['train_y_std']:.4f}  "
          f"(compare to new season before trusting predictions)")

    return model, meta, device




# =============================================================================
# APPLY TO NEW DATA
# =============================================================================

def apply_mlp(model, meta, device, predictors_new, log_predictor_eps=1e-30):
    """
    Apply a loaded MLP to a new set of predictors.

    Parameters
    ----------
    model           : MLP in eval mode (from load_mlp)
    meta            : dict from load_mlp
    device          : str
    predictors_new  : list of np.ndarray, each shape (n_events,)
                      Same order and meaning as at training time.
    log_predictor_eps : float — same epsilon used at training time

    Returns
    -------
    y_pred : np.ndarray, shape (n_events,) — predictions in physical units
    """
    feat_scaler   = meta['feat_scaler']
    target_scaler = meta['target_scaler']
    log_pred_mask = meta['log_pred_mask']

    X_new = np.column_stack(predictors_new).astype(np.float32)

    # apply the same per-predictor log transforms used at training time
    X_tr = X_new.copy()
    if log_pred_mask.any():
        X_tr[:, log_pred_mask] = np.log(
            np.abs(X_tr[:, log_pred_mask]) + log_predictor_eps
        )

    X_scaled = feat_scaler.transform(X_tr).astype(np.float32)
    X_t      = torch.from_numpy(X_scaled).to(device)

    model.eval()
    with torch.no_grad():
        out = model(X_t).cpu().numpy()

    out_clipped = np.clip(out, -8.0, 8.0)   # matches _infer in fit_mlp_general
    log_y_pred  = target_scaler.inverse_transform(
        out_clipped.reshape(-1, 1)
    ).flatten()

    return np.exp(log_y_pred)
