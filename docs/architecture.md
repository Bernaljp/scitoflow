# Architecture

This page lays the method bare: the latent variables, the cascade drift, the integration
scheme, every loss term, and a parameter inventory. It is the page to read if you want to
know exactly what the model computes.

## The state cascade

scIToFlow models each tissue spot as a point on a shared latent dynamical trajectory. The
default `full` topology is the cascade

$$
r \;\rightarrow\; c \;\rightarrow\; u \;\rightarrow\; s \;\rightarrow\; r ,
$$

where $c$ is chromatin, $u$ is unspliced RNA, $s$ is spliced RNA, and $r$ is a latent
**regulation** node that closes the loop back onto chromatin. Each non-regulatory state has
its own latent block $z_c, z_u, z_s \in \mathbb{R}^{L}$ (default $L = 20$); the regulation
node is $z_r \in \mathbb{R}^{d_r}$ (default $d_r = 2$). A spatial **niche factor**
$h \in \mathbb{R}^{2 d_h}$ conditions the drift. The cascade is configurable
({doc}`api` / {func}`scitoflow.get_topology`): `no_unspliced` drops $u$, `rna_only` and
`minimal` drop chromatin, and `labeling` replaces $u,s$ with nascent/total for
metabolic-labeling data.

## Encoders and the niche factor

- **Per-state encoders.** One MLP per cascade state maps its observed modality (the
  moment-smoothed layer $M_c, M_u, M_s$) to a Gaussian posterior over its latent block.
- **Latent time.** An encoder on the concatenated latent state produces a per-spot latent
  time $t \in [0, t_{\max}]$ (a sigmoid of an encoded logit, scaled by a learned
  $t_{\max}$). Latent time is a **dynamical-state coordinate**, not calendar age.
- **Niche factor $h$.** Two single-graph GraphSAGE encoders, one over the **spatial**
  neighbor graph and one over the **expression** neighbor graph, each produce a $d_h$-vector;
  their concatenation is $h$. Setting `use_spatial=False` zeros the spatial half (the niche
  ablation / counterfactual); `use_expr_gnn=False` replaces the expression encoder with an
  MLP.

## The drift and the regulatory loop

The latent dynamics are a neural ODE. For each non-regulatory state, a small MLP produces
its drift from its parent's latent, its own latent, and (optionally) time:

$$
\dot z_{\text{node}} = f_{\text{node}}\big(z_{\text{parent}},\, z_{\text{node}}\,[,\, t]\big).
$$

The regulatory drift is a function of the terminal RNA latent, the regulation node, and the
niche factor,

$$
\dot z_r = f_{r}\big(z_{s},\, z_r,\, h\big),
$$

and $z_r$ feeds back onto the first cascade state (chromatin), closing the loop. Setting
`use_feedback=False` opens the loop (the feedback-edge ablation).

## Integration: the grid-ODE (default)

Integrating a separate trajectory per spot is $O(N^2)$ in the batch. Instead, scIToFlow
integrates **one** batched trajectory over a fixed $K$-point time grid and places each spot
on it by differentiable linear interpolation at its own latent time (a veloVI/RegVelo-style
trick). This is $O(K \cdot N)$ in memory and about an order of magnitude faster, and matches
the per-cell integration to roughly $10^{-10}$. Set `use_grid_ode=False` to recover the
legacy per-cell integration; `use_adjoint=True` uses `torchdiffeq`'s adjoint (slower here,
so off by default).

## Loss terms

The per-spot objective (see {meth}`scitoflow.VAE.loss`) sums:

```{list-table}
:header-rows: 1
:widths: 26 74

* - Term
  - What it does
* - **Reconstruction**
  - Gaussian log-likelihood (learned per-gene scale) of each decoded modality, decoded from
    *both* the data-encoded latent and the trajectory latent at time $t$.
* - **Trajectory consistency**
  - Ties the trajectory latent to the data-encoded latent for the decoded states, so the ODE
    solution explains the encoded state.
* - **KL**
  - Gaussian KL on the decoded-state latent posteriors plus a down-weighted KL on latent
    time, with a warmup schedule (`kl_warmup_steps`, `kl_final_weight`).
* - **Tangent velocity**
  - Consistency of the latent velocity with the expression-graph tangent space (GraphVelo
    style), weighted by `tangent_reg_weight`. This is where velocity enters as a regularizer.
* - **Biophysical positive-rate**
  - Optional (`latent_reg=True`) penalty enforcing a positive production rate on the terminal
    production edge (splicing in `full`).
* - **Time prior**
  - Optional (`time_prior_weight>0`) soft supervision of latent time by a known label
    (developmental stage or labeling time): a lag-tolerant smooth-L1 (`mode="soft"`) or a
    within-batch ordering hinge (`mode="rank"`). Deliberately soft, so asynchronous cells are
    allowed.
```

## Parameter inventory

Counts for a reference configuration (`observed=100`, `latent_dim=20`, `zr_dim=2`,
`h_dim=2`, all hidden widths `25`, 2-layer encoders/decoders, `topology="full"`,
`use_spatial=True`, `use_expr_gnn=True`, `use_grid_ode=True`): **30,304** trainable
parameters, distributed as

```{list-table}
:header-rows: 1
:widths: 34 16 12 38

* - Component
  - Params
  - Share
  - Role
* - Per-state encoders (×3)
  - 12,645
  - 41.7%
  - modality → latent posterior (4,215 each)
* - Decoders (×2 decoded: u, s)
  - 7,550
  - 24.9%
  - latent → gene space (3,775 each)
* - Velocity field
  - 4,912
  - 16.2%
  - cascade drift MLPs (4,185) + regulatory drift (727)
* - Basis decoder
  - 2,955
  - 9.8%
  - tangent-space velocity basis
* - Time encoder
  - 1,577
  - 5.2%
  - latent state → latent time
* - Spatial GraphSAGE (h)
  - 242
  - 0.8%
  - spatial niche factor
* - Expression GraphSAGE (h)
  - 242
  - 0.8%
  - expression niche factor
* - Direct parameters
  - 181
  - 0.6%
  - `max_time`, `initial_z`, `theta`, `theta_z`
```

The count scales with the number of genes (encoders/decoders) and the latent width; smaller
topologies (`no_unspliced`, `rna_only`, `minimal`) drop the corresponding encoders/decoders.
You can reproduce the table for any configuration:

```python
import torch; torch.set_default_dtype(torch.float64)
from scitoflow import VAE
m = VAE(observed=100, latent_dim=20, use_expr_gnn=True, topology="full")
total = sum(p.numel() for p in m.parameters())
by_component = {name: sum(p.numel() for p in child.parameters())
                for name, child in m.named_children()}
print(total, by_component)
```

## Readouts

After fitting, {meth}`scitoflow.VAE.reconstruct_latent` returns, for every spot, the
data-encoded latent, the full trajectory latent, the latent **velocity**, the inferred
**latent time**, and the **niche factor** $h$. {meth}`scitoflow.VAE.project_velocities`
projects latent velocities onto any 2D embedding through the learned basis, ready for
streamline plotting. See the {doc}`notebooks/03_training_and_readouts` tutorial.
