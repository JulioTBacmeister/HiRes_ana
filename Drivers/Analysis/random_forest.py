#####################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline




def build_predictor_matrix(
    field_arrays,
    field_names,
    zlev,
    z_top_km=12.0,
    t_indices=[-3, -2, -1],   # t-2, t-1, t0 (last 3 of 7 steps)
    mode='profile',            # 'profile' = centre only, 'full' = profile + spatial stats
    ):
    """
    Build a compact predictor matrix from 4D event columns.
    
    Parameters
    ----------
    field_arrays : list of np.ndarray, shape (n_events, n_t, n_z, n_y, n_x)
                   or                   shape (n_events, n_t, n_y, n_x) for 2D fields
    field_names  : list of str
    zlev         : np.ndarray, shape (n_z,) — z levels in km
    z_top_km     : float — top of predictor vertical range in km
    t_indices    : list of int — which time indices to include (negative = from end)
    mode         : 'profile' — vertical profile at patch centre only
                   'full'    — profile + patch mean + centre anomaly
    
    Returns
    -------
    X : np.ndarray, shape (n_events, n_features)
        Predictor matrix.
    feature_names : list of str
        Name of each feature column — essential for interpretation.
    """
    # vertical mask: surface to z_top_km
    z_mask  = zlev <= z_top_km
    z_levels = zlev[z_mask]
    n_z_use  = z_mask.sum()

    n_events = field_arrays[0].shape[0]
    n_t_full = field_arrays[0].shape[1]

    # resolve negative time indices
    t_idx = [i % n_t_full for i in t_indices]
    t_labels = [f't{i - n_t_full + 1}' if i < n_t_full - 1 else 't0'
                for i in t_idx]

    feature_blocks = []
    feature_names  = []

    for arr, name in zip(field_arrays, field_names):
        is_2D = arr.ndim == 4   # (n_events, n_t, n_y, n_x)
        n_y   = arr.shape[-2]
        n_x   = arr.shape[-1]
        cy, cx = n_y // 2, n_x // 2

        for ti, tl in zip(t_idx, t_labels):

            if is_2D:
                # --- 2D field (e.g. rain) -----------------------------------
                centre_val = arr[:, ti, cy, cx]          # (n_events,)
                feature_blocks.append(centre_val[:, None])
                feature_names.append(f'{name}|{tl}|centre')

                if mode == 'full':
                    patch_mean = arr[:, ti, :, :].mean(axis=(-2, -1))
                    feature_blocks.append(patch_mean[:, None])
                    feature_names.append(f'{name}|{tl}|patch_mean')

                    anomaly = centre_val - patch_mean
                    feature_blocks.append(anomaly[:, None])
                    feature_names.append(f'{name}|{tl}|centre_anomaly')

            else:
                # --- 3D field (n_events, n_t, n_z, n_y, n_x) ---------------

                # vertical profile at patch centre
                profile = arr[:, ti, :, cy, cx]          # (n_events, n_z)
                profile_tropo = profile[:, z_mask]        # (n_events, n_z_use)
                feature_blocks.append(profile_tropo)
                for z in z_levels:
                    feature_names.append(f'{name}|{tl}|z={z:.1f}km|centre')

                if mode == 'full':
                    # patch mean profile
                    #patch_mean = arr[:, ti, :, :, :].mean(axis=(-2, -1))  # (n_events, n_z)
                    patch_mean = arr[:, ti, :, 0:5, 0:].mean(axis=(-2, -1))  # (n_events, n_z)
                    patch_mean_tropo = patch_mean[:, z_mask]
                    feature_blocks.append(patch_mean_tropo)
                    for z in z_levels:
                        feature_names.append(f'{name}|{tl}|z={z:.1f}km|patch_mean')

                    # centre anomaly profile
                    anomaly = profile_tropo - patch_mean_tropo
                    feature_blocks.append(anomaly)
                    for z in z_levels:
                        feature_names.append(f'{name}|{tl}|z={z:.1f}km|anomaly')

    X = np.concatenate(feature_blocks, axis=1)

    print(f"Predictor matrix: {n_events} events x {X.shape[1]} features")
    print(f"  {len(field_arrays)} fields, {len(t_idx)} timesteps, "
          f"{n_z_use} z levels (surface to {z_top_km}km)")

    return X, feature_names


def build_target(epwp_4D, zlev, z_target_km=15.0):
    """
    Extract target variable: epwp at z_target_km, patch centre, t0.
    """
    z_idx = np.argmin(np.abs(zlev - z_target_km))
    n_y, n_x = epwp_4D.shape[-2], epwp_4D.shape[-1]
    cy, cx   = n_y // 2, n_x // 2
    #y = epwp_4D[:, -1, z_idx, cy, cx]   # t0 = last time index
    y = np.mean( epwp_4D[:, :, z_idx, 0:, 0:], axis=(1,2,3) )   # t0 = last time index
    print(f"Target: epwp at z={zlev[z_idx]:.1f}km, centre, t0")
    print(f"  shape: {y.shape},  mean: {y.mean():.4f},  std: {y.std():.4f}")
    return y


def fit_random_forest(X, y, test_size=0.2, n_estimators=200, random_state=42):
    """
    Fit a Random Forest regressor and report performance.

    Returns
    -------
    rf       : fitted RandomForestRegressor
    splits   : dict with X_train, X_test, y_train, y_test
    scores   : dict with train/test R2 and RMSE
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred_train = rf.predict(X_train)
    y_pred_test  = rf.predict(X_test)

    scores = {
        'r2_train':   r2_score(y_train, y_pred_train),
        'r2_test':    r2_score(y_test,  y_pred_test),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'rmse_test':  np.sqrt(mean_squared_error(y_test,  y_pred_test)),
    }

    print(f"Train R²: {scores['r2_train']:.3f}   RMSE: {scores['rmse_train']:.4f}")
    print(f"Test  R²: {scores['r2_test']:.3f}   RMSE: {scores['rmse_test']:.4f}")

    splits = dict(X_train=X_train, X_test=X_test,
                  y_train=y_train, y_test=y_test,
                  y_pred_test=y_pred_test)

    return rf, splits, scores


def plot_feature_importance(rf, feature_names, top_n=30):
    """
    Plot top_n most important features by mean decrease in impurity.
    """
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, top_n * 0.3 + 1))
    ax.barh(range(top_n), importances[idx][::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=8)
    ax.set_xlabel('Feature importance (mean decrease impurity)')
    ax.set_title(f'Top {top_n} predictors')
    plt.tight_layout()
    plt.show()

    return idx, importances



##########################################################################
#   Simplified work flow
##########################################################################
def fit_rf_general(
    predictors,
    predictor_names,
    target,
    event_times=None,        # NEW: time index for each event
    train_interval=None,     # NEW: (t_start, t_end) for training
    test_interval=None,      # NEW: (t_start, t_end) for testing
    test_size=0.2,
    n_estimators=200,
    min_samples_leaf=5,
    max_depth=None,
    max_features='sqrt',
    stratify_labels=None,
    random_state=42,
    scale_predictors=True,   # NEW — default True
):
    """
    General Random Forest regression workflow.
    
    Parameters
    ----------
    predictors : list of np.ndarray, each shape (n_events,)
    predictor_names : list of str
    target : np.ndarray, shape (n_events,)
    event_times : np.ndarray or None, shape (n_events,)
        Time index for each event (e.g. 0-248).
        Required if train_interval or test_interval are provided.
    train_interval : tuple (t_start, t_end) or None
        If provided, use events with t_start <= t < t_end for training.
        Overrides test_size/stratify_labels.
    test_interval : tuple (t_start, t_end) or None
        If provided, use events with t_start <= t < t_end for testing.
        If None but train_interval is set, uses all events not in training.
    test_size : float
        Used only if train_interval is None (random split mode).
    n_estimators : int
    min_samples_leaf : int
    max_depth : int or None
    max_features : str or float
    stratify_labels : np.ndarray or None
        Used only if train_interval is None (random split mode).
    random_state : int
    
    Returns
    -------
    rf      : fitted RandomForestRegressor
    results : dict with splits, scores, importances
    """
    assert len(predictors) == len(predictor_names), \
        "predictors and predictor_names must have the same length"
    assert all(p.shape == target.shape for p in predictors), \
        "all predictors and target must have shape (n_events,)"

    # --- assemble feature matrix ------------------------------------------
    X = np.column_stack(predictors)
    y = target
    print(f"Feature matrix: {X.shape[0]} events x {X.shape[1]} predictors")

    # --- train/test split --------------------------------------------------
    if train_interval is not None:
        # --- time-based split ----------------------------------------------
        assert event_times is not None, \
            "event_times must be provided when using train_interval"
        
        t_train_start, t_train_end = train_interval
        train_mask = (event_times >= t_train_start) & \
                     (event_times <  t_train_end)

        if test_interval is not None:
            t_test_start, t_test_end = test_interval
            test_mask = (event_times >= t_test_start) & \
                        (event_times <  t_test_end)
        else:
            # use everything outside the training interval
            test_mask = ~train_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        print(f"Time-based split:")
        print(f"  Train: t=[{t_train_start}, {t_train_end})  "
              f"→ {train_mask.sum()} events")
        if test_interval is not None:
            print(f"  Test:  t=[{t_test_start},  {t_test_end})   "
                  f"→ {test_mask.sum()} events")
            gap_start = t_train_end
            gap_end   = t_test_start
            if gap_end > gap_start:
                print(f"  Gap:   t=[{gap_start}, {gap_end})  "
                      f"({gap_end - gap_start} timesteps = "
                      f"{(gap_end - gap_start)*3:.0f} hours)")
        else:
            print(f"  Test:  all other events → {test_mask.sum()} events")

    else:
        # --- random split (original behaviour) ----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
        print(f"Random split: {len(y_train)} train, {len(y_test)} test events")

# --- fit ---------------------------------------------------------------
    _rf = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )

    if scale_predictors:
        rf = Pipeline([
            ('scaler', StandardScaler()),
            ('rf',     _rf)
        ])
    else:
        rf = _rf

    rf.fit(X_train, y_train)

    # --- scores ------------------------------------------------------------
    y_pred_train = rf.predict(X_train)
    y_pred_test  = rf.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test  = r2_score(y_test,  y_pred_test)
    r_test, p_test = stats.pearsonr(y_test, y_pred_test)

    print(f"Train R²:  {r2_train:.3f}")
    print(f"Test  R²:  {r2_test:.3f}")
    print(f"Test  r:   {r_test:.3f}  (p={p_test:.2e})")

    # --- cross validation (only in random split mode) ----------------------
    cv_scores = None
    if train_interval is None:
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
        print(f"5-fold CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # --- feature importances -----------------------------------------------
    # handle Pipeline vs plain RF
    if scale_predictors:
        importances = rf['rf'].feature_importances_
    else:
        importances = rf.feature_importances_

    importance_order = np.argsort(importances)[::-1]

    # --- results dict ------------------------------------------------------
    results = {
        'X_train':          X_train,
        'X_test':           X_test,
        'y_train':          y_train,
        'y_test':           y_test,
        'y_pred_train':     y_pred_train,
        'y_pred_test':      y_pred_test,
        'r2_train':         r2_train,
        'r2_test':          r2_test,
        'r_test':           r_test,
        'p_test':           p_test,
        'cv_scores':        cv_scores,
        'importances':      importances,
        'importance_order': importance_order,
        'predictor_names':  predictor_names,
        'split_mode':       'time' if train_interval else 'random',
        'scale_predictors': scale_predictors,
    }

    return rf, results


def fit_rf_general_sv(
    predictors,
    predictor_names,
    target,
    event_times=None,        # NEW: time index for each event
    train_interval=None,     # NEW: (t_start, t_end) for training
    test_interval=None,      # NEW: (t_start, t_end) for testing
    test_size=0.2,
    n_estimators=200,
    min_samples_leaf=5,
    max_depth=None,
    max_features='sqrt',
    stratify_labels=None,
    random_state=42,
):
    """
    General Random Forest regression workflow.
    
    Parameters
    ----------
    predictors : list of np.ndarray, each shape (n_events,)
    predictor_names : list of str
    target : np.ndarray, shape (n_events,)
    event_times : np.ndarray or None, shape (n_events,)
        Time index for each event (e.g. 0-248).
        Required if train_interval or test_interval are provided.
    train_interval : tuple (t_start, t_end) or None
        If provided, use events with t_start <= t < t_end for training.
        Overrides test_size/stratify_labels.
    test_interval : tuple (t_start, t_end) or None
        If provided, use events with t_start <= t < t_end for testing.
        If None but train_interval is set, uses all events not in training.
    test_size : float
        Used only if train_interval is None (random split mode).
    n_estimators : int
    min_samples_leaf : int
    max_depth : int or None
    max_features : str or float
    stratify_labels : np.ndarray or None
        Used only if train_interval is None (random split mode).
    random_state : int
    
    Returns
    -------
    rf      : fitted RandomForestRegressor
    results : dict with splits, scores, importances
    """
    assert len(predictors) == len(predictor_names), \
        "predictors and predictor_names must have the same length"
    assert all(p.shape == target.shape for p in predictors), \
        "all predictors and target must have shape (n_events,)"

    # --- assemble feature matrix ------------------------------------------
    X = np.column_stack(predictors)
    y = target
    print(f"Feature matrix: {X.shape[0]} events x {X.shape[1]} predictors")

    # --- train/test split --------------------------------------------------
    if train_interval is not None:
        # --- time-based split ----------------------------------------------
        assert event_times is not None, \
            "event_times must be provided when using train_interval"
        
        t_train_start, t_train_end = train_interval
        train_mask = (event_times >= t_train_start) & \
                     (event_times <  t_train_end)

        if test_interval is not None:
            t_test_start, t_test_end = test_interval
            test_mask = (event_times >= t_test_start) & \
                        (event_times <  t_test_end)
        else:
            # use everything outside the training interval
            test_mask = ~train_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        print(f"Time-based split:")
        print(f"  Train: t=[{t_train_start}, {t_train_end})  "
              f"→ {train_mask.sum()} events")
        if test_interval is not None:
            print(f"  Test:  t=[{t_test_start},  {t_test_end})   "
                  f"→ {test_mask.sum()} events")
            gap_start = t_train_end
            gap_end   = t_test_start
            if gap_end > gap_start:
                print(f"  Gap:   t=[{gap_start}, {gap_end})  "
                      f"({gap_end - gap_start} timesteps = "
                      f"{(gap_end - gap_start)*3:.0f} hours)")
        else:
            print(f"  Test:  all other events → {test_mask.sum()} events")

    else:
        # --- random split (original behaviour) ----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
        print(f"Random split: {len(y_train)} train, {len(y_test)} test events")

    # --- fit ---------------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # --- scores ------------------------------------------------------------
    y_pred_train = rf.predict(X_train)
    y_pred_test  = rf.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test  = r2_score(y_test,  y_pred_test)
    r_test, p_test = stats.pearsonr(y_test, y_pred_test)

    print(f"Train R²:  {r2_train:.3f}")
    print(f"Test  R²:  {r2_test:.3f}")
    print(f"Test  r:   {r_test:.3f}  (p={p_test:.2e})")

    # --- cross validation (only in random split mode) ----------------------
    cv_scores = None
    if train_interval is None:
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
        print(f"5-fold CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # --- feature importances -----------------------------------------------
    importances      = rf.feature_importances_
    importance_order = np.argsort(importances)[::-1]

    # --- results dict ------------------------------------------------------
    results = {
        'X_train':          X_train,
        'X_test':           X_test,
        'y_train':          y_train,
        'y_test':           y_test,
        'y_pred_train':     y_pred_train,
        'y_pred_test':      y_pred_test,
        'r2_train':         r2_train,
        'r2_test':          r2_test,
        'r_test':           r_test,
        'p_test':           p_test,
        'cv_scores':        cv_scores,
        'importances':      importances,
        'importance_order': importance_order,
        'predictor_names':  predictor_names,
        'split_mode':       'time' if train_interval else 'random',
    }

    return rf, results



def plot_rf_results(results, top_n=None):
    """
    Plot predicted vs actual and feature importances.
    
    Parameters
    ----------
    results : dict as returned by fit_rf_general
    top_n : int or None
        How many top features to show. None = show all.
    """
    predictor_names  = results['predictor_names']
    n_predictors     = len(predictor_names)
    top_n            = top_n or n_predictors
    top_n            = min(top_n, n_predictors)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, top_n * 0.35 + 1)))

    # --- predicted vs actual -----------------------------------------------
    ax = axes[0]
    ax.scatter(results['y_test'], results['y_pred_test'],
               alpha=0.5, s=20, color='steelblue')
    lims = [
        min(results['y_test'].min(), results['y_pred_test'].min()),
        max(results['y_test'].max(), results['y_pred_test'].max()),
    ]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Predicted vs Actual (test)\n'
                 f'R²={results["r2_test"]:.3f}  '
                 f'r={results["r_test"]:.3f}')

    # --- feature importances -----------------------------------------------
    ax = axes[1]
    idx  = results['importance_order'][:top_n]
    vals = results['importances'][idx][::-1]
    nms  = [predictor_names[i] for i in idx[::-1]]

    ax.barh(range(top_n), vals, color='steelblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(nms, fontsize=8)
    ax.set_xlabel('Feature importance (mean decrease impurity)')
    ax.set_title(f'Top {top_n} predictors')

    plt.tight_layout()
    plt.show()


def add_permutation_importance(rf, results, n_repeats=20):
    """
    Compute and plot permutation importance on test set.
    Adds results to the results dict in place.
    """
    perm = permutation_importance(
        rf,
        results['X_test'],
        results['y_test'],
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )

    results['perm_importances_mean'] = perm.importances_mean
    results['perm_importances_std']  = perm.importances_std

    # plot
    predictor_names = results['predictor_names']
    n_predictors    = len(predictor_names)
    idx  = np.argsort(perm.importances_mean)[::-1]
    vals = perm.importances_mean[idx][::-1]
    errs = perm.importances_std[idx][::-1]
    nms  = [predictor_names[i] for i in idx[::-1]]

    fig, ax = plt.subplots(figsize=(8, max(4, n_predictors * 0.35 + 1)))
    ax.barh(range(n_predictors), vals, xerr=errs, color='steelblue')
    ax.set_yticks(range(n_predictors))
    ax.set_yticklabels(nms, fontsize=8)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('Permutation importance')
    ax.set_title('Permutation importance (test set)')
    plt.tight_layout()
    plt.show()

    return results




