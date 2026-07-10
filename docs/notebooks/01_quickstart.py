# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quickstart
#
# This notebook fits scIToFlow end to end in a few minutes, on CPU, using the built-in
# {func}`scitoflow.simulate_dataset`. You will:
#
# 1. simulate a small spatial multi-omic dataset,
# 2. build and train the `full` model, and
# 3. read out the inferred **spatial latent time** and compare it to the ground truth.
#
# ```{note}
# The model runs in double precision, so we set the default dtype to `float64` once at the
# top. Every input tensor you pass to the model should be `float64` as well.
# ```

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.grid": False})
NICHE_COLORS = ["#E69F00", "#56B4E9", "#009E73"]  # Okabe-Ito, colorblind-safe

# %% [markdown]
# ## 1. Simulate a dataset
#
# `simulate_dataset` returns an `AnnData` with the moment-smoothed layers the model
# consumes (`M_c` chromatin, `M_u` unspliced, `M_s` spliced), spatial coordinates, and a
# ground-truth `latent_time` and `niche` label we can validate against.

# %%
from scitoflow import simulate_dataset

adata = simulate_dataset(n_genes=50, grid=20, n_niches=3, seed=0)
adata

# %% [markdown]
# The spots lie on a 20x20 grid. Latent time increases along the diagonal, and three spatial
# niches band the tissue. Here is the ground truth we will try to recover, plus one gene's
# spliced signal.

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
x, y = adata.obs["x_position"], adata.obs["y_position"]

sc0 = axes[0].scatter(x, y, c=adata.obs["latent_time"], cmap="viridis", s=45, marker="s")
axes[0].set_title("ground-truth latent time")
plt.colorbar(sc0, ax=axes[0], shrink=0.8)

niche = adata.obs["niche"].astype(int).values
axes[1].scatter(x, y, c=[NICHE_COLORS[i] for i in niche], s=45, marker="s")
axes[1].set_title("spatial niches")

g = adata.var_names[0]
sc2 = axes[2].scatter(x, y, c=adata[:, g].layers["M_s"].toarray().ravel(),
                      cmap="magma", s=45, marker="s")
axes[2].set_title(f"spliced ({g})")
plt.colorbar(sc2, ax=axes[2], shrink=0.8)

for ax in axes:
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()

# %% [markdown]
# ## 2. Build and train the model
#
# We build a small `full` model (chromatin -> unspliced -> spliced -> regulation) with the
# dual spatial + expression GraphSAGE niche encoder, and train for a handful of epochs.
# {func}`scitoflow.train_vae` picks the device automatically (CPU here).

# %%
from scitoflow import VAE, train_vae

model = VAE(
    observed=adata.n_vars, latent_dim=8, zr_dim=2, h_dim=2,
    encoder_hidden=20, decoder_hidden=20, t_encoder_hidden=20, graph_hidden=20,
    velocity_model_hidden=20, num_steps=30, ode_grid=30,
    topology="full", use_spatial=True, use_feedback=True,
    use_grid_ode=True, use_expr_gnn=True,
)
print(f"{sum(p.numel() for p in model.parameters()):,} trainable parameters")
print("cascade states:", model.topo.states)

# %%
epochs, val_recon, val_traj, edge_spatial, adj_expr = train_vae(
    model=model, adata=adata, epochs=25, batch_size=128, learning_rate=5e-3,
    tangent_loss_params={"a": 1.0, "b": 10.0, "reg_lambda": 1.0},
    checkpoint_folder="/tmp/scitoflow_quickstart",
)

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.plot(epochs, val_recon, "-o", ms=3, label="reconstruction (test)")
ax.plot(epochs, val_traj, "-o", ms=3, label="trajectory (test)")
ax.set_xlabel("epoch"); ax.set_ylabel("validation MSE"); ax.legend(); ax.set_title("training history")
fig.tight_layout()

# %% [markdown]
# ## 3. Read out the inferred latent time
#
# {meth}`scitoflow.VAE.reconstruct_latent` returns, for every spot, the latent trajectory,
# the latent velocity, the inferred latent time, and the niche factor. Because we used the
# dual GNN (`use_expr_gnn=True`), we pass both the spatial and the expression graph. The
# expression neighbor list from training is converted to an edge index with the model helper.

# %%
model.eval()
data = {st: torch.tensor(adata.layers[model.topo.layer[st]].toarray(), dtype=torch.float64)
        for st in model.topo.states}
edge_expr = model._adj_to_edge_index(adj_expr.cpu())

with torch.no_grad():
    _, latent_full, velocity, latent_time, h = model.reconstruct_latent(
        data, edge_index_spatial=edge_spatial.cpu(), edge_index_expression=edge_expr)

t_inferred = latent_time.cpu().numpy()
t_true = adata.obs["latent_time"].values
rho = spearmanr(t_inferred, t_true).statistic
print(f"Spearman(inferred latent time, ground truth) = {rho:.3f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
axes[0].scatter(t_true, t_inferred, s=14, alpha=0.5, color="#0072B2")
axes[0].set_xlabel("ground-truth latent time"); axes[0].set_ylabel("inferred latent time")
axes[0].set_title(f"Spearman rho = {rho:.2f}")

sc = axes[1].scatter(x, y, c=t_inferred, cmap="viridis", s=45, marker="s")
axes[1].set_title("inferred latent time (spatial)")
axes[1].set_aspect("equal"); axes[1].set_xticks([]); axes[1].set_yticks([])
plt.colorbar(sc, ax=axes[1], shrink=0.8)
fig.tight_layout()

# %% [markdown]
# In only a few epochs the inferred latent time recovers the spatial gradient. Latent time
# is defined up to an orientation, so a strong monotone (Spearman) relationship, positive or
# negative, is the thing to look for, not a 45-degree line.
#
# ## Next
#
# - {doc}`02_modality_topologies` - run the same model on subsets of the modalities.
# - {doc}`03_training_and_readouts` - the niche factor, velocity streamlines, a within-model
#   niche counterfactual, and the optional time prior.
