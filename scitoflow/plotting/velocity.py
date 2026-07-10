"""
Streamline plotting of projected 2D velocities on an embedding.

FAITHFUL first-pass extraction from the method-development notebook
(Base/model1new_multivelo_organized.ipynb), Phase A1 consolidation.
Logic preserved verbatim; only module imports were added/normalized.
NOT yet unit-tested or made device-agnostic (hardcoded .cuda() remains) -
that is the research-software hardening pass. See PLAN.md Phase A.
"""

# --- faithful extraction: notebook cell 68 (model1new_multivelo_organized.ipynb) ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Union, Dict

def plot_streamline_from_vectors(
    ax: Axes,
    X_emb: np.ndarray,
    V_emb: np.ndarray,
    color_points_by: Optional[np.ndarray] = None,
    cmap: str = 'viridis',
    scatter_s: float = 5,
    scatter_alpha: float = 0.3,
    grid_density: float = 1.0,
    smooth: float = 0.5,
    n_neighbors: Optional[int] = None,
    min_mass: float = 1.0,
    cutoff_perc: float = 5.0,
    stream_density: float = 2.0,
    stream_linewidth: float = 1.0,
    stream_color: str = 'black',
    stream_arrowsize: float = 1.0,
    show_colorbar: bool = False,  # <-- NEW
    colorbar_label: str = '',   # <-- NEW
    colorbar_kwargs: Optional[Dict] = None, # <-- NEW
    show_axes: bool = True      # <-- NEW
) -> Axes:
    """
    Generates a streamline plot from 2D coordinates and 2D velocities.
    
    (Docstring content from previous function...)
    """
    
    # --- 1. Compute Velocity on Grid (Unchanged) ---
    
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]
    if color_points_by is not None:
        color_points_by = color_points_by[idx_valid]

    n_obs, n_dim = X_emb.shape
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    
    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * grid_density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid_flat = np.vstack([i.flat for i in meshes_tuple]).T

    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid_flat)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = norm.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid_flat = (V_emb[neighs] * weight[:, :, None]).sum(1)
    V_grid_flat /= np.maximum(1, p_mass)[:, None]

    X_grid = np.stack([np.unique(X_grid_flat[:, 0]), np.unique(X_grid_flat[:, 1])])
    ns = int(np.sqrt(len(V_grid_flat[:, 0])))
    V_grid = V_grid_flat.T.reshape(2, ns, ns)

    mass = np.sqrt((V_grid**2).sum(0))
    min_mass_val = 10**(min_mass - 6)
    min_mass_val = np.clip(min_mass_val, None, np.max(mass) * 0.9)
    cutoff = mass.reshape(V_grid[0].shape) < min_mass_val

    if cutoff_perc is not None:
        length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)

    V_grid[0][cutoff] = np.nan

    # --- 2. Plot Scatter (cells) ---
    # Capture the scatter plot object
    scatter_plot = ax.scatter(
        X_emb[:, 0],
        X_emb[:, 1],
        c=color_points_by,
        cmap=cmap,
        s=scatter_s,
        alpha=scatter_alpha,
        zorder=0,
        rasterized=True
    )

    # --- 3. Plot Streamlines (Unchanged) ---
    lengths = np.sqrt((V_grid**2).sum(0))
    valid_lengths = lengths[~np.isnan(lengths)]
    if len(valid_lengths) > 0:
        stream_lw = stream_linewidth * 2 * lengths / valid_lengths.max()
    else:
        stream_lw = stream_linewidth

    ax.streamplot(
        X_grid[0],
        X_grid[1],
        V_grid[0],
        V_grid[1],
        density=stream_density,
        color=stream_color,
        linewidth=stream_lw,
        arrowsize=stream_arrowsize,
        zorder=3
    )
    
    # --- 4. Add Colorbar and Tweak Axes (NEW) ---
    
    if show_colorbar and color_points_by is not None:
        cbar_kwargs = colorbar_kwargs or {}
        # Add some sensible defaults that can be overridden
        if 'shrink' not in cbar_kwargs:
            cbar_kwargs['shrink'] = 0.7
        if 'aspect' not in cbar_kwargs:
            cbar_kwargs['aspect'] = 20
        
        # Get the figure from the axis to add the colorbar
        fig = ax.get_figure()
        cbar = fig.colorbar(scatter_plot, ax=ax, **cbar_kwargs)
        
        if colorbar_label:
            cbar.set_label(colorbar_label)

    if not show_axes:
        ax.axis('off')

    return ax

