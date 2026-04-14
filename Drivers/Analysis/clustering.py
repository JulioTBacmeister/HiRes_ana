import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

def assemble_feature_matrix(field_arrays, field_names=None):
    """
    Assemble separate field arrays into a 2D feature matrix for PCA.
    
    Parameters
    ----------
    field_arrays : list of np.ndarray, each shape (n_events, n_t, n_z, n_y, n_x)
        One array per field, in any order.
    field_names : list of str or None
        Names for each field, e.g. ['vort', 'u', 'v', 'mf_abs'].
        Used for labelling only.
    
    Returns
    -------
    X : np.ndarray, shape (n_events, n_features)
        Feature matrix ready for PCA. Each row is one event,
        each column is one flattened spatiotemporal feature.
    feature_info : dict
        Metadata about the feature matrix.
    """
    if field_names is None:
        field_names = [f'field_{i}' for i in range(len(field_arrays))]
    
    n_events = field_arrays[0].shape[0]
    
    # sanity check all arrays have same shape
    for name, arr in zip(field_names, field_arrays):
        assert arr.shape[0] == n_events, f"{name} has inconsistent n_events"
        assert arr.shape == field_arrays[0].shape, f"{name} has inconsistent shape"
    
    n_t, n_z, n_y, n_x = field_arrays[0].shape[1:]
    n_per_field = n_t * n_z * n_y * n_x
    
    # flatten each field and concatenate along feature axis
    field_blocks = []
    for arr in field_arrays:
        field_blocks.append(arr.reshape(n_events, -1))  # (n_events, n_per_field)
    
    X = np.concatenate(field_blocks, axis=1)  # (n_events, n_fields * n_per_field)
    
    feature_info = {
        'field_names':  field_names,
        'n_per_field':  n_per_field,
        'n_fields':     len(field_arrays),
        'n_features':   X.shape[1],
        'n_events':     n_events,
        'field_shape':  (n_t, n_z, n_y, n_x),
    }
    
    print(f"Feature matrix assembled: {n_events} events x {X.shape[1]} features")
    print(f"  {len(field_arrays)} fields x {n_per_field} values each")
    
    return X, feature_info


def run_pca(X, n_components=50, normalize=True):
    """
    Normalize and run PCA on the feature matrix.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_events, n_features)
    n_components : int
        Number of PCA components to retain.
    normalize : bool
        If True, standardize each feature to zero mean and unit variance
        before PCA. Important when fields have very different magnitudes.
    
    Returns
    -------
    X_pca : np.ndarray, shape (n_events, n_components)
        PCA-projected data.
    pca : fitted sklearn PCA object
    scaler : fitted sklearn StandardScaler (or None if normalize=False)
    """
    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = None
        X_scaled = X
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # report explained variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    for threshold in [0.5, 0.75, 0.9]:
        n_needed = np.searchsorted(cum_var, threshold) + 1
        print(f"  {threshold*100:.0f}% variance explained by {n_needed} components")
    
    return X_pca, pca, scaler


def plot_pca_variance(pca):
    """Scree plot to help choose number of components."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].bar(range(1, len(pca.explained_variance_ratio_)+1),
                pca.explained_variance_ratio_)
    axes[0].set_xlabel('Component')
    axes[0].set_ylabel('Explained variance ratio')
    axes[0].set_title('Scree plot')
    
    axes[1].plot(range(1, len(pca.explained_variance_ratio_)+1),
                 np.cumsum(pca.explained_variance_ratio_), marker='o', ms=3)
    axes[1].axhline(0.9, color='r', linestyle='--', label='90%')
    axes[1].axhline(0.75, color='orange', linestyle='--', label='75%')
    axes[1].set_xlabel('Number of components')
    axes[1].set_ylabel('Cumulative explained variance')
    axes[1].set_title('Cumulative variance')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def run_clustering(X_pca, n_clusters=2, method='both'):
    """
    Cluster events in PCA space.
    
    Parameters
    ----------
    X_pca : np.ndarray, shape (n_events, n_components)
    n_clusters : int
        Number of clusters for k-means.
    method : 'kmeans', 'hierarchical', or 'both'
    
    Returns
    -------
    labels : dict with keys 'kmeans' and/or 'hierarchical'
        Cluster label (0-indexed) for each event.
    linkage_matrix : np.ndarray or None
        Linkage matrix from hierarchical clustering, for dendrogram plotting.
    """
    labels = {}
    linkage_matrix = None
    
    if method in ('kmeans', 'both'):
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels['kmeans'] = km.fit_predict(X_pca)
        print(f"K-means cluster sizes: "
              f"{[np.sum(labels['kmeans']==k) for k in range(n_clusters)]}")
    
    if method in ('hierarchical', 'both'):
        linkage_matrix = linkage(X_pca, method='ward')
        labels['hierarchical'] = fcluster(linkage_matrix,
                                          t=n_clusters,
                                          criterion='maxclust') - 1  # 0-indexed
        print(f"Hierarchical cluster sizes: "
              f"{[np.sum(labels['hierarchical']==k) for k in range(n_clusters)]}")
    
    return labels, linkage_matrix


def plot_dendrogram(linkage_matrix, n_events, truncate=True):
    """Plot hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(12, 4))
    dendrogram(linkage_matrix,
               ax=ax,
               truncate_mode='lastp' if truncate else None,
               p=30,  # show last 30 merges
               leaf_rotation=90,
               leaf_font_size=8)
    ax.set_title('Hierarchical clustering dendrogram')
    ax.set_xlabel('Event index (or cluster size)')
    ax.set_ylabel('Ward distance')
    plt.tight_layout()
    plt.show()


def plot_pca_scatter(X_pca, labels, label_key='kmeans'):
    """Scatter plot of first two PCA components, coloured by cluster."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cluster_labels = labels[label_key]
    n_clusters = len(np.unique(cluster_labels))
    for k in range(n_clusters):
        mask = cluster_labels == k
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f'Cluster {k}  (n={mask.sum()})',
                   alpha=0.7, s=40)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'PCA scatter — coloured by {label_key} cluster')
    ax.legend()
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_cluster_composites(
    field_4D,
    cluster_labels,
    field_name,
    zlev,
    z_idx,
    t_labels=None,
    cmap='RdBu_r',
    figsize_per_panel=(3, 3),
    suptitle=None,
    vmax=None,
):
    """
    Plot per-cluster composites as a 2D grid: rows=clusters, columns=time steps.
    
    Parameters
    ----------
    field_4D : np.ndarray, shape (n_events, n_t, n_z, n_y, n_x)
        Field data for all events.
    cluster_labels : np.ndarray, shape (n_events,)
        Cluster index for each event (0-indexed).
    field_name : str
        Name of the field for titles/labels.
    zlev : np.ndarray, shape (n_z,)
        Array of z levels in km (or whatever unit you use).
    z_idx : int
        Index into the z dimension to plot.
    t_labels : list of str or None
        Labels for each time step, e.g. ['t-6','t-5',...,'t0'].
        If None, auto-generated as t-n ... t0.
    cmap : str
        Colormap. RdBu_r is good for vorticity and wind.
    figsize_per_panel : tuple
        Size of each individual (y,x) panel in inches.
    suptitle : str or None
        Overall figure title.
    vmax : float or None
        Color scale limit. If None, computed symmetrically from data.
        Shared across all clusters and times for direct comparison.
    
    Returns
    -------
    fig : matplotlib Figure
    axes : np.ndarray of Axes, shape (n_clusters, n_t)
    composites : np.ndarray, shape (n_clusters, n_t, n_y, n_x)
        The composite fields, useful for further analysis.
    """
    n_events, n_t, n_z, n_y, n_x = field_4D.shape
    cluster_ids = np.unique(cluster_labels)
    n_clusters  = len(cluster_ids)

    if t_labels is None:
        t_labels = [f't{i - n_t + 1}' if i < n_t - 1 else 't0'
                    for i in range(n_t)]

    # --- compute composites ------------------------------------------------
    composites = np.zeros((n_clusters, n_t, n_y, n_x))
    counts     = np.zeros(n_clusters, dtype=int)
    for k, cid in enumerate(cluster_ids):
        mask           = cluster_labels == cid
        counts[k]      = mask.sum()
        composites[k]  = field_4D[mask, :, z_idx, :, :].mean(axis=0)

    # --- color scale -------------------------------------------------------
    if vmax is None:
        absmax = np.nanmax(np.abs(composites))
        vmax   = absmax
    vmin = -vmax

    # --- figure layout -----------------------------------------------------
    fig_w = figsize_per_panel[0] * n_t + 1.5       # +1.5 for row labels
    fig_h = figsize_per_panel[1] * n_clusters + 1.0  # +1.0 for title
    fig   = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(
        n_clusters, n_t,
        figure=fig,
        hspace=0.35,
        wspace=0.15,
    )
    axes = np.empty((n_clusters, n_t), dtype=object)

    # --- plot --------------------------------------------------------------
    z_label = f'{zlev[z_idx]:.0f} km'

    for k, cid in enumerate(cluster_ids):
        for ti in range(n_t):
            ax = fig.add_subplot(gs[k, ti])
            axes[k, ti] = ax

            im = ax.pcolormesh(
                composites[k, ti],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            # mark centre of patch
            ax.axhline(n_y // 2, color='k', lw=0.5, ls='--', alpha=0.4)
            ax.axvline(n_x // 2, color='k', lw=0.5, ls='--', alpha=0.4)

            ax.set_xticks([])
            ax.set_yticks([])

            # column headers (time labels) on top row only
            if k == 0:
                ax.set_title(t_labels[ti], fontsize=9)

            # row labels on left column only
            if ti == 0:
                ax.set_ylabel(
                    f'Cluster {cid}\n(n={counts[k]})',
                    fontsize=9,
                    rotation=0,
                    labelpad=55,
                    va='center',
                )

    # --- shared colorbar ---------------------------------------------------
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=field_name)

    # --- title -------------------------------------------------------------
    title = suptitle or f'{field_name} composite at z={z_label}'
    fig.suptitle(title, fontsize=12, y=1.01)

    #plt.show()
    return fig, axes, composites