"""
Spatial overlay plotting on tissue images; training-history plot.

FAITHFUL first-pass extraction from the method-development notebook
(Base/model1new_multivelo_organized.ipynb), Phase A1 consolidation.
Logic preserved verbatim; only module imports were added/normalized.
NOT yet unit-tested or made device-agnostic (hardcoded .cuda() remains) -
that is the research-software hardening pass. See PLAN.md Phase A.
"""

# --- faithful extraction: notebook cell 9 (model1new_multivelo_organized.ipynb) ---
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import matplotlib

def plot_obs_spatial(
    adata,
    image_path,
    obs='latent_time',
    figsize=(15, 5),
    spot_size=30,
    rotate=0,
    flip=None,
    layer='spliced',
    cmap='viridis',
    axes = None,
):
    """
    Plots a specified observation over the spatial tissue image using a scatter plot.

    Tries to find `obs` first in adata.obs, else in adata.var.index with data 
    extracted from adata.layers[layer].

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    image_path : str
        Path to tissue background image.
    obs : str
        Observation key in adata.obs or gene name in adata.var.index.
    figsize : tuple
        Figure size.
    spot_size : int
        Scatter spot size.
    rotate : int
        Additional rotation of image and coordinates: 0,90,180,270 degrees.
    flip : str or None
        Additional flip: None, 'ud' (vertical), or 'lr' (horizontal).
    layer : str
        Layer name to use if `obs` found in adata.var.index (default 'spliced').
    cmap : str or Colormap, optional
        Colormap for scatter plot (default 'viridis').
    """

    tissue_img = mpimg.imread(image_path)
    tissue_img = np.rot90(tissue_img, 1)
    tissue_img = np.flipud(tissue_img)

    x_coords = np.array(adata.obs['pxl_col_in_fullres'])
    y_coords = np.array(adata.obs['pxl_row_in_fullres'])

    if obs in adata.obs.columns:
        obs_values = adata.obs[obs].values
    elif obs in adata.var_names:
        if layer not in adata.layers:
            print(f"⚠️ Layer '{layer}' not found in adata.layers.")
            return
        try:
            gene_idx = adata.var_names.get_loc(obs)
            obs_layer = adata.layers[layer]
            if hasattr(obs_layer, "toarray"):
                obs_values = obs_layer[:, gene_idx].toarray().ravel()
            else:
                obs_values = obs_layer[:, gene_idx]
        except Exception as e:
            print(f"⚠️ Could not extract '{obs}' from layer '{layer}': {e}")
            return
    else:
        print(f"⚠️ Observation '{obs}' not found in adata.obs or adata.var_names.")
        return

    H, W = tissue_img.shape[:2]
    trans = Affine2D()
    if rotate in {90, 180, 270}:
        trans.rotate_deg_around(W/2, H/2, rotate)
    elif rotate != 0:
        print("⚠️ Only 0, 90, 180, 270 rotations supported.")
    if flip == 'ud':
        trans.scale(1, -1).translate(0, H)
    elif flip == 'lr':
        trans.scale(-1, 1).translate(W, 0)
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        plt.tight_layout(pad=3.0)

        axes[0].imshow(tissue_img, transform=trans + axes[0].transData)
        axes[0].set_title('Tissue Background')
        axes[0].axis('off')

        axes[1].imshow(tissue_img, transform=trans + axes[1].transData)
        scatter = axes[1].scatter(
            x_coords, y_coords, s=spot_size, c=obs_values,
            cmap=cmap, alpha=0.8, transform=trans + axes[1].transData
        )
        axes[1].set_title(f"{obs.replace('_', ' ').capitalize()} Overlay")
        axes[1].axis('off')
        cbar = plt.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label(f"{obs.replace('_', ' ').capitalize()} Value")

        axes[2].set_facecolor('black')
        axes[2].scatter(
            x_coords, y_coords, s=spot_size, c=obs_values,
            cmap=cmap, transform=trans + axes[2].transData
        )
        axes[2].set_title(f"{obs.replace('_', ' ').capitalize()} Distribution")
        axes[2].set_aspect('equal', adjustable='box')
        axes[2].invert_yaxis()
        axes[2].axis('off')

        plt.show()
    elif isinstance(axes, matplotlib.axes.Axes):
        # Single axes, not in a list or array
        ax = axes
        ax.imshow(tissue_img, transform=trans + ax.transData)
        scatter = ax.scatter(
            x_coords, y_coords, s=spot_size, c=obs_values,
            cmap=cmap, alpha=0.8, transform=trans + ax.transData
        )
        ax.set_title(f"{obs.replace('_', ' ').capitalize()} Overlay")
        ax.axis('off')
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{obs.replace('_', ' ').capitalize()} Value")

    elif len(axes) == 1:
        ax = axes if not isinstance(axes, (list, np.ndarray)) else axes[0]
        ax.imshow(tissue_img, transform=trans + ax.transData)
        scatter = ax.scatter(
            x_coords, y_coords, s=spot_size, c=obs_values,
            cmap=cmap, alpha=0.8, transform=trans + ax.transData
        )
        ax.set_title(f"{obs.replace('_', ' ').capitalize()} Overlay")
        ax.axis('off')
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{obs.replace('_', ' ').capitalize()} Value")


    elif len(axes) == 2:
        ax_img, ax_overlay = axes

        # Image alone
        ax_img.imshow(tissue_img, transform=trans + ax_img.transData)
        ax_img.set_title('Tissue Background')
        ax_img.axis('off')

        # Image with overlay
        ax_overlay.imshow(tissue_img, transform=trans + ax_overlay.transData)
        scatter = ax_overlay.scatter(
            x_coords, y_coords, s=spot_size, c=obs_values,
            cmap=cmap, alpha=0.8, transform=trans + ax_overlay.transData
        )
        ax_overlay.set_title(f"{obs.replace('_', ' ').capitalize()} Overlay")
        ax_overlay.axis('off')
        cbar = plt.colorbar(scatter, ax=ax_overlay, fraction=0.046, pad=0.04)
        cbar.set_label(f"{obs.replace('_', ' ').capitalize()} Value")


    else:  # 3 or more axes
        # Use the original behavior for 3 axes
        axes[0].imshow(tissue_img, transform=trans + axes[0].transData)
        axes[0].set_title('Tissue Background')
        axes[0].axis('off')

        axes[1].imshow(tissue_img, transform=trans + axes[1].transData)
        scatter = axes[1].scatter(
            x_coords, y_coords, s=spot_size, c=obs_values,
            cmap=cmap, alpha=0.8, transform=trans + axes[1].transData
        )
        axes[1].set_title(f"{obs.replace('_', ' ').capitalize()} Overlay")
        axes[1].axis('off')
        cbar = plt.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label(f"{obs.replace('_', ' ').capitalize()} Value")

        axes[2].set_facecolor('black')
        axes[2].scatter(
            x_coords, y_coords, s=spot_size, c=obs_values,
            cmap=cmap, transform=trans + axes[2].transData
        )
        axes[2].set_title(f"{obs.replace('_', ' ').capitalize()} Distribution")
        axes[2].set_aspect('equal', adjustable='box')
        axes[2].invert_yaxis()
        axes[2].axis('off')

# --- faithful extraction: notebook cell 10 (model1new_multivelo_organized.ipynb) ---
def plot_history(epoch_history, val_ae_history, val_traj_history):
    """Plot training history for model losses."""
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot epoch history
    axs[0].plot(epoch_history)
    axs[0].set_title('Epoch Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    
    # Plot validation AE history
    axs[1].plot(val_ae_history)
    axs[1].set_title('Validation AE Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    
    # Plot validation trajectory history
    axs[2].plot(val_traj_history)
    axs[2].set_title('Validation Trajectory Loss')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

