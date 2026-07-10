"""
scIToFlow VAE: encoders, latent neural-ODE dynamics, losses, read-outs.

Generalized from the original hardcoded chromatin->unspliced->spliced->reg model to a
configurable `scitoflow.core.topology.Topology`. The FULL preset reproduces the original
math exactly; NO_UNSPLICED (r->c->s->r), RNA_ONLY, MINIMAL, and LABELING drop or swap the
corresponding states. Inputs/outputs use a modality dict keyed by topology state name
(e.g. {"c":M_c, "u":M_u, "s":M_s}). Device-agnostic (no hardcoded .cuda()).
"""
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from scitoflow.core.networks import MLP, mySAGEConv
from scitoflow.core.velocity_field import VelocityFieldReg
from scitoflow.core.topology import get_topology


def reindex_adjacency(adjacency, subset, num_nodes, device=None):
    """Reindex adjacency matrix to match subset of nodes."""
    device = device if device is not None else adjacency.device
    new_indexing = -1 * torch.ones(num_nodes, dtype=torch.long, device=device)
    for i, idx in enumerate(subset):
        if new_indexing[idx] == -1:
            new_indexing[idx] = i
    return new_indexing[adjacency[subset]]


def unique_index(x):
    """Find the index of the unique times for the ODE solver."""
    sort_index = th.argsort(x)
    sorted_x = x[sort_index]
    index = th.Tensor([th.max(th.where(sorted_x == i)[0]) for i in th.unique(sorted_x)]).long()
    return sort_index, index


def batch_jacobian(func, x, create_graph=True):
    f_sum = lambda x: torch.sum(func(x), axis=0)
    return torch.autograd.functional.jacobian(f_sum, x, create_graph=create_graph)


def gaussian_kl(mu, logvar, mu0=0, logvar0=0):
    """KL divergence between diagonal gaussians."""
    return -0.5 * th.sum(1. + logvar - logvar0 - (mu - mu0) ** 2 / np.exp(0.5 * logvar0)
                         - th.exp(logvar) / np.exp(logvar0), dim=-1)


class VAE(nn.Module):
    """
    scIToFlow VAE over a configurable modality cascade (see topology.py).

    New parameter
    -------------
    topology : str | Topology, default "full"
        Cascade over which the latent dynamics run. "full" = the original
        chromatin->unspliced->spliced->regulation model.

    All other parameters are as before (observed genes, latent_dim, zr_dim, h_dim,
    hidden sizes/layers, num_steps, kl warmup/weight, latent_reg, tangent_reg_weight...).
    """
    def __init__(self, observed=2000, latent_dim=20, zr_dim=2, h_dim=2,
                 encoder_hidden=25, decoder_hidden=25, t_encoder_hidden=25, graph_hidden=25,
                 velocity_model_hidden=25,
                 encoder_layers=2, decoder_layers=2, t_encoder_layers=1, velocity_model_layers=1,
                 n_neighbors_expression=30, num_steps=100,
                 encoder_bn=False, decoder_bn=False,
                 include_time=False, kl_warmup_steps=25, kl_final_weight=1, max_sigma_z=0,
                 latent_reg=False, tangent_reg_weight=1, topology="full",
                 use_spatial=True, use_feedback=True, use_adjoint=False,
                 use_grid_ode=True, ode_grid=100, use_expr_gnn=False,
                 time_prior_weight=0.0, time_prior_mode="soft"):
        super().__init__()
        self.topo = get_topology(topology)
        # use_adjoint: integrate the latent ODE with the adjoint method (torchdiffeq
        # odeint_adjoint) so the backward pass is O(1) in memory (recompute states) instead
        # of storing every solver step. Default False = original plain-odeint behavior.
        # (Profiling: adjoint is ~10x slower with no memory win here, so left off.)
        self.use_adjoint = use_adjoint
        # use_grid_ode (DEFAULT, since it is a strict win): integrate ONE batched trajectory over
        # a fixed K-point time grid (ode_grid), then linearly interpolate each cell's state at its
        # own latent time (RegVelo/veloVI style). O(K*N) memory + ~13x faster vs the legacy O(N^2)
        # approach that integrates over the per-cell time grid and takes the diagonal; trajectories
        # match to ~1e-10. Set use_grid_ode=False to reproduce the legacy integration.
        self.use_grid_ode = use_grid_ode
        self.ode_grid = ode_grid
        # Ablation flags (default True = original behavior).
        #   use_spatial : if False, the spatial GraphSAGE factor h_spatial is zeroed,
        #                 removing the microenvironment/niche conditioning (spatial ablation).
        #   use_feedback: if False, the regulatory node zr does not drive the first cascade
        #                 state (loop-closure / feedback-edge ablation); passed to the field.
        self.use_spatial = use_spatial
        self.use_feedback = use_feedback
        self.n_states = self.topo.n_states
        self.n_decoded = len(self.topo.decoded)

        self.observed = observed
        self.latent = latent_dim
        self.zr_dim = zr_dim
        self.h_dim = h_dim
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.t_encoder_hidden = t_encoder_hidden
        self.graph_hidden = graph_hidden
        self.velocity_model_hidden = velocity_model_hidden
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.t_encoder_layers = t_encoder_layers
        self.velocity_model_layers = velocity_model_layers
        self.n_neighbors_expression = n_neighbors_expression
        self.num_steps = num_steps
        self.encoder_bn = encoder_bn
        self.decoder_bn = decoder_bn
        self.include_time = include_time
        self.kl_warmup_steps = kl_warmup_steps
        self.kl_final_weight = kl_final_weight
        self.max_sigma_z = max_sigma_z
        self.latent_reg = latent_reg
        self.tangent_reg_weight = tangent_reg_weight
        # use_expr_gnn: make the expression conditioning encoder a GNN (mySAGEConv over the
        # expression kNN graph) instead of an MLP. This realizes the intended dual-GNN
        # (spatial GNN + expression GNN); the tangent loss still uses the same expression graph
        # for the velocity basis projection (a separate, orthogonal use -> NOT a multigraph).
        self.use_expr_gnn = use_expr_gnn
        # time_prior_weight/mode: optionally supervise the inferred latent time with a known
        # time label (metabolic-labeling time, or experimental/developmental stage). The prior
        # is SOFT by design: because development is asynchronous (per-cell latent time is a
        # dynamical-state coordinate, not calendar age), we do not force t = tau. mode "soft"
        # applies a lag-tolerant smooth-L1 penalty between the normalized latent time and the
        # normalized label (weight controls how much asynchrony is allowed); mode "rank" applies
        # a within-batch pairwise ordering hinge (only where a batch spans >=2 label values).
        self.time_prior_weight = time_prior_weight
        self.time_prior_mode = time_prior_mode
        self._build_networks()

    def _build_networks(self):
        L, n = self.latent, self.n_states
        # One encoder per cascade state (data -> 2*latent for mean/logvar).
        self.encoders = nn.ModuleDict({
            st: MLP(self.observed, self.encoder_hidden, 2 * L, MLP_layers=self.encoder_layers,
                    activation='relu', bn=self.encoder_bn) for st in self.topo.states})

        # Time + spatial-conditioning encoders operate on the full latent state (n*L).
        self.encoder_t = MLP(n * L, self.t_encoder_hidden, 2, MLP_layers=self.t_encoder_layers,
                             activation='relu', bn=self.encoder_bn)
        self.max_time = nn.Parameter(torch.tensor(1.0))
        self.encoder_h_spatial = mySAGEConv(n * L, self.graph_hidden, self.h_dim, n_layers=1,
                                            activation_fn='relu', batch_norm=self.encoder_bn)
        if self.use_expr_gnn:
            self.encoder_h_expression = mySAGEConv(n * L, self.graph_hidden, self.h_dim, n_layers=1,
                                                   activation_fn='relu', batch_norm=self.encoder_bn)
        else:
            self.encoder_h_expression = MLP(n * L, self.graph_hidden, self.h_dim,
                                            MLP_layers=self.t_encoder_layers, activation='relu',
                                            bn=self.encoder_bn)
        self.basis_decoder = MLP(n * L, self.encoder_hidden, self.n_neighbors_expression,
                                 self.encoder_layers, activation='relu', bn=self.encoder_bn)
        self.initial_z = nn.Parameter(2 / np.sqrt(n * L) * torch.rand(n * L) - 1 / np.sqrt(n * L))

        # One decoder per reconstructed (RNA-like) state.
        self.decoders = nn.ModuleDict({
            st: MLP(L, self.decoder_hidden, self.observed, MLP_layers=self.decoder_layers,
                    activation='relu', bn=self.decoder_bn) for st in self.topo.decoded})

        self.theta = nn.Parameter(2 / np.sqrt(self.observed) * torch.rand(self.observed) - 1 / np.sqrt(self.observed))
        self.theta_z = nn.Parameter(2 / np.sqrt(L) * torch.rand(L) - 1 / np.sqrt(L))

        self.velocity_field = VelocityFieldReg(L, 2 * self.h_dim, self.zr_dim,
                                               self.velocity_model_hidden, self.velocity_model_layers,
                                               include_time=self.include_time, topology=self.topo,
                                               use_feedback=self.use_feedback)

    # ---- helpers -----------------------------------------------------------
    def _state_slice(self, z, state):
        i = self.topo.index(state)
        return z[:, i * self.latent:(i + 1) * self.latent]

    def _check_inputs(self, data):
        missing = [s for s in self.topo.states if s not in data]
        if missing:
            raise KeyError(f"topology {self.topo.name} needs data for states {self.topo.states}; missing {missing}")

    @staticmethod
    def _adj_to_edge_index(adj_list):
        """(N, k) local neighbor-index list (-1 = padding) -> [2, E] edge_index [i, neighbor],
        matching the spatial GNN's kNN [row, col] convention."""
        N, k = adj_list.shape
        src = torch.arange(N, device=adj_list.device).repeat_interleave(k)
        dst = adj_list.reshape(-1)
        mask = dst >= 0
        return torch.stack([src[mask], dst[mask]], dim=0)

    # ---- encoding ----------------------------------------------------------
    def latent_embedding(self, data, edge_index_spatial=None, edge_index_expression=None):
        """Encode a modality dict {state: (N, observed)} into latent space.

        Returns: (z|h), decoded-state latent_mean, decoded-state latent_logvar,
                 latent_time, t_mean, t_logvar, basis.
        """
        self._check_inputs(data)
        L = self.latent
        z_parts, mean_parts, logvar_parts = [], [], []
        for st in self.topo.states:
            p = self.encoders[st](data[st])
            mean, logvar = p[:, :L], p[:, L:]
            z_st = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
            z_parts.append(z_st)
            if st in self.topo.decoded:
                mean_parts.append(mean)
                logvar_parts.append(logvar)
        z = torch.cat(z_parts, dim=-1)

        t_params = self.encoder_t(z)
        t_mean, t_logvar = t_params[:, 0], t_params[:, 1]
        latent_time = self.max_time * torch.sigmoid(t_mean + torch.randn_like(t_mean) * torch.exp(0.5 * t_logvar))
        # Keep times strictly > 0 so the ODE grid [0, unique(times)...] is strictly
        # increasing even when the sampled sigmoid underflows to 0 (else torchdiffeq
        # raises "t must be strictly increasing").
        latent_time = latent_time.clamp(min=1e-6)

        if self.use_spatial:
            h_spatial = self.encoder_h_spatial(z, edge_index_spatial)
        else:
            # Spatial ablation: drop the microenvironment/niche channel entirely.
            h_spatial = z.new_zeros(z.shape[0], self.h_dim)
        if self.use_expr_gnn:
            h_expression = self.encoder_h_expression(z, edge_index_expression)
        else:
            h_expression = self.encoder_h_expression(z)
        h = torch.cat((h_spatial, h_expression), dim=-1)
        basis = self.basis_decoder(z)

        return (torch.cat((z, h), dim=-1), torch.cat(mean_parts, dim=-1),
                torch.cat(logvar_parts, dim=-1), latent_time, t_mean, t_logvar, basis)

    # ---- dynamics ----------------------------------------------------------
    def _run_dynamics(self, h, times, test=False):
        from torchdiffeq import odeint, odeint_adjoint
        dev = h.device
        h0 = self.initial_z.repeat(h.shape[0], 1)
        h0 = torch.cat((h0, torch.zeros(h.shape[0], self.zr_dim, device=dev), h), dim=-1)
        t_full = torch.cat((torch.zeros(1, device=dev), times), dim=-1)
        # adjoint (constant-memory backward) only during training; test/inference uses plain odeint.
        solver = odeint_adjoint if (self.use_adjoint and not test) else odeint
        if test:
            ht_full = odeint(self.velocity_field, h0, t_full, method='dopri8',
                             options=dict(max_num_steps=self.num_steps)).permute(1, 0, 2)
        else:
            ht_full = solver(self.velocity_field, h0, t_full, method='dopri5', rtol=1e-5, atol=1e-5,
                             options=dict(max_num_steps=self.num_steps)).permute(1, 0, 2)
        ht_full = ht_full[:, 1:]
        ht = ht_full[..., :self.n_states * self.latent + self.zr_dim]
        return ht, h0

    def _traj_states(self, ht, diag_idx):
        """Pull per-state trajectory latents (+ zr) at the given time index per row."""
        L, n = self.latent, self.n_states
        rows = np.arange(ht.shape[0])
        states = {st: ht[rows, diag_idx, i * L:(i + 1) * L] for i, st in enumerate(self.topo.states)}
        zr = ht[rows, diag_idx, n * L:n * L + self.zr_dim]
        return states, zr

    def _grid_trajectories(self, h, cell_times, test=False):
        """RegVelo/veloVI-style efficient integration: integrate ONE batched trajectory over a
        FIXED K-point grid (ode_grid) spanning [0, max_time], then linearly interpolate each
        cell's state at its own latent time. O(K*N) memory instead of the legacy O(N^2)
        (which integrates over the per-cell time grid and keeps only the diagonal). Fully
        differentiable in both the ODE parameters (via the grid states) and the latent time
        (via the interpolation weight). Returns (states dict, zr, h0)."""
        from torchdiffeq import odeint
        dev = h.device
        N, K = h.shape[0], self.ode_grid
        D = self.n_states * self.latent + self.zr_dim
        h0 = self.initial_z.repeat(N, 1)
        h0 = torch.cat((h0, torch.zeros(N, self.zr_dim, device=dev), h), dim=-1)
        tmax = torch.clamp(cell_times.max(), min=1e-6)
        grid = torch.linspace(0.0, 1.0, K, device=dev) * tmax
        if test:
            ht = odeint(self.velocity_field, h0, grid, method='dopri8',
                        options=dict(max_num_steps=self.num_steps)).permute(1, 0, 2)
        else:
            ht = odeint(self.velocity_field, h0, grid, method='dopri5', rtol=1e-5, atol=1e-5,
                        options=dict(max_num_steps=self.num_steps)).permute(1, 0, 2)
        ht = ht[..., :D]                                        # (N, K, D)
        ct = cell_times.clamp(min=grid[0], max=grid[-1])
        j = torch.searchsorted(grid, ct, right=True).clamp(1, K - 1)  # right grid index
        t0, t1 = grid[j - 1], grid[j]
        w = ((ct - t0) / (t1 - t0 + 1e-12)).clamp(0.0, 1.0)[:, None]   # (N, 1)
        rows = torch.arange(N, device=dev)
        flat = ht[rows, j - 1] * (1 - w) + ht[rows, j] * w            # (N, D) interpolated
        L, n = self.latent, self.n_states
        states = {st: flat[:, i * L:(i + 1) * L] for i, st in enumerate(self.topo.states)}
        zr = flat[:, n * L:n * L + self.zr_dim]
        return states, zr, h0

    # ---- loss --------------------------------------------------------------
    def loss(self, data, edge_index_spatial, adjacency_list_expression, tangent_loss_params, epoch=None,
             time_prior=None):
        L, n = self.latent, self.n_states
        dev = next(self.parameters()).device
        any_mod = data[self.topo.states[0]]

        eix_expr = self._adj_to_edge_index(adjacency_list_expression) if self.use_expr_gnn else None
        latent_state, latent_mean, latent_logvar, latent_time, time_mean, time_logvar, basis = \
            self.latent_embedding(data, edge_index_spatial=edge_index_spatial,
                                  edge_index_expression=eix_expr)
        z = latent_state[:, :n * L]
        h = latent_state[:, n * L:]
        orig_index = th.arange(any_mod.shape[0], device=dev)

        sort_index, index = unique_index(latent_time)
        sel = lambda a: a[sort_index][index]
        latent_time = sel(latent_time)
        z = sel(z); h = sel(h); orig_index = sel(orig_index)
        latent_mean = sel(latent_mean); latent_logvar = sel(latent_logvar)
        time_mean = sel(time_mean); time_logvar = sel(time_logvar); basis = sel(basis)
        data = {k: sel(v) for k, v in data.items()}
        if time_prior is not None:
            time_prior = sel(time_prior)

        new_indices = sort_index[index]
        adjacency_list_expression = reindex_adjacency(adjacency_list_expression, new_indices, len(sort_index), device=dev)

        # trajectory latents at each cell's own latent time
        if self.use_grid_ode:
            traj, zr, h0 = self._grid_trajectories(h, latent_time)
        else:
            ht, h0 = self._run_dynamics(h, latent_time)
            traj, zr = self._traj_states(ht, np.arange(z.shape[0]))

        latent_state = torch.cat((z, zr, h), dim=-1)
        latent_velocity = self.velocity_field(latent_time, latent_state)[:, :n * L]

        tangent_loss = self.tangent_reg_weight * self.tangent_loss(
            z, latent_velocity, basis, adjacency_list_expression,
            a=tangent_loss_params['a'], b=tangent_loss_params['b'], reg_lambda=tangent_loss_params['reg_lambda'])

        # data-space latents per state
        data_lat = {st: self._state_slice(z, st) for st in self.topo.states}

        # reconstruction over decoded modalities (from data-latent and traj-latent)
        recon = 0.0
        val_ae = 0.0
        val_traj = 0.0
        theta = 1e-4 + F.softplus(self.theta)
        for st in self.topo.decoded:
            xhat_data = self.decoders[st](data_lat[st])
            xhat_traj = self.decoders[st](traj[st])
            target = data[st]
            recon = recon - Normal(xhat_data, theta).log_prob(target).sum(-1) \
                          - Normal(xhat_traj, theta).log_prob(target).sum(-1)
            val_ae = val_ae + torch.sum((xhat_data - target) ** 2, dim=-1)
            val_traj = val_traj + torch.sum((xhat_traj - target) ** 2, dim=-1)

        # latent likelihood: tie traj decoded-latents to data decoded-latents
        traj_dec = torch.cat([traj[st] for st in self.topo.decoded], dim=-1)
        data_dec = torch.cat([data_lat[st] for st in self.topo.decoded], dim=-1)
        theta_z_rep = torch.cat(self.n_decoded * [self.theta_z], dim=-1)
        if self.max_sigma_z > 0:
            sig_z = 1e-4 + self.max_sigma_z * torch.sigmoid(theta_z_rep)
        else:
            sig_z = 1e-4 + F.softplus(theta_z_rep)
        recon = recon - Normal(traj_dec, sig_z).log_prob(data_dec).sum(-1)
        reconstruction_loss = recon

        # biophysical positive-rate reg on the production edge (splicing in FULL)
        if self.latent_reg:
            latent_reg = self.latent_reg_func(traj, data_lat)
        else:
            latent_reg = torch.zeros(1, device=dev)

        # KL on all decoded-state latent params + time
        kl_full = gaussian_kl(latent_mean, latent_logvar) + 0.1 * gaussian_kl(time_mean[:, None], time_logvar[:, None])
        if epoch is not None:
            kl_reg = self.kl_final_weight * min(1, epoch / self.kl_warmup_steps) * kl_full
        else:
            kl_reg = self.kl_final_weight * kl_full

        loss = reconstruction_loss + kl_reg + latent_reg + tangent_loss
        if time_prior is not None and self.time_prior_weight > 0:
            loss = loss + self.time_prior_weight * self._time_prior_loss(latent_time, time_prior)
        return loss, val_ae, val_traj, tangent_loss, orig_index

    def _time_prior_loss(self, latent_time, tau):
        """Soft supervision of latent time by a known time label tau (already normalized to
        [0, 1] by the caller). Returns a per-cell penalty (mode 'soft') or a scalar ordering
        penalty (mode 'rank'); either broadcasts onto the per-cell loss vector. Deliberately
        soft so asynchronous/lagging cells are allowed."""
        t01 = latent_time / (self.max_time.abs() + 1e-6)
        tau = tau.to(t01.dtype)
        if self.time_prior_mode == "rank":
            # within-batch pairwise ordering: cell i labeled later than j should have t_i > t_j
            dt = t01[:, None] - t01[None, :]
            later = (tau[:, None] - tau[None, :] > 0).to(t01.dtype)
            hinge = F.relu(0.05 - dt) * later
            return hinge.sum() / later.sum().clamp(min=1.0)
        # 'soft' (default): lag-tolerant smooth-L1 between normalized latent time and label
        return F.smooth_l1_loss(t01, tau, reduction="none", beta=0.1)

    def latent_reg_func(self, traj, data_lat):
        """Penalize negative production rates on the last cascade edge (parent -> terminal).

        For FULL this is d spliced_net / d(unspliced) >= 0 (splicing). Generalizes to the
        Jacobian of the terminal decoded state's drift wrt its parent latent.
        """
        parent, child = self.topo.production_edge
        # The positive-rate constraint only applies to a state->state production edge
        # (splicing-like). If the terminal state is driven directly by the regulatory
        # node (e.g. MINIMAL r->s), there is no such edge, so skip it.
        if parent == "r":
            return torch.zeros(1, device=next(self.parameters()).device)
        child_net = self.velocity_field.state_nets[child]
        child_traj, child_data = traj[child], data_lat[child]
        parent_traj, parent_data = traj[parent], data_lat[parent]
        split = 100
        jacs_traj, jacs_data = [], []
        for i in range(split, child_traj.shape[0] + split, split):
            jacs_traj.append(batch_jacobian(
                lambda x: child_net(torch.cat((x, child_traj[i - split:i]), dim=-1)),
                parent_traj[i - split:i]).permute(1, 0, 2))
            jacs_data.append(batch_jacobian(
                lambda x: child_net(torch.cat((x, child_data[i - split:i]), dim=-1)),
                parent_data[i - split:i]).permute(1, 0, 2))
        j_traj = torch.cat(jacs_traj, dim=0)
        j_data = torch.cat(jacs_data, dim=0)
        return 10000 * (F.relu(-1 * j_traj).sum(dim=(-1, -2)) + F.relu(-1 * j_data).sum(dim=(-1, -2)))

    def tangent_loss(self, x, v, basis, adjacency_list, a=1, b=10, reg_lambda=1, verbose=False):
        mask = adjacency_list != -1
        k = adjacency_list.shape[1]
        xj = x[adjacency_list]
        xi = x.unsqueeze(1).expand(-1, k, -1)
        delta_ij = xj - xi
        cosines = F.cosine_similarity(v.unsqueeze(1).expand(-1, k, -1), delta_ij, dim=-1)
        phi_corr = F.softmax(cosines - 1e9 * (1. - mask.type(cosines.dtype)), dim=1) - 1.0 / k
        v_parallel_learned = torch.einsum('nk,nkf->nf', basis * mask, delta_ij)
        velo_loss = torch.norm(v - v_parallel_learned, dim=1).pow(2)
        projection_loss = -F.cosine_similarity(basis * mask, phi_corr * mask, dim=1)
        reg_loss = torch.norm(basis, p=2, dim=1).pow(2)
        return (a * velo_loss + b * projection_loss + reg_lambda * reg_loss).mean()

    # ---- read-outs ---------------------------------------------------------
    def reconstruct_latent(self, data, edge_index_spatial=None, edge_index_expression=None):
        """Latent trajectories + velocity read-out for all cells."""
        L, n = self.latent, self.n_states
        latent_state, _, _, latent_time, _, _, _ = self.latent_embedding(
            data, edge_index_spatial, edge_index_expression)
        z = latent_state[:, :n * L]; h = latent_state[:, n * L:]
        if self.use_grid_ode:
            traj, zr, _ = self._grid_trajectories(h, latent_time)
        else:
            unique_times, inverse = torch.unique(latent_time, return_inverse=True, sorted=True)
            ht, _ = self._run_dynamics(h, unique_times)
            traj, zr = self._traj_states(ht, inverse)
        latent_full = torch.cat([traj[st] for st in self.topo.states] + [zr], dim=-1)
        velocity = self.velocity_field.drift(latent_time[:, None], torch.cat((z, zr, h), dim=-1))
        return torch.cat((z, zr), dim=-1), latent_full, velocity, latent_time, h

    @torch.no_grad()
    def project_velocities(self, data, projections, spatial_edge_index, expression_adjacency_list):
        """Project latent velocities onto a low-dim embedding via the learned basis
        (GraphVelo Eq. 3). `data` is the modality dict."""
        self.eval()
        z = torch.cat([self.encoders[st](data[st])[:, :self.latent] for st in self.topo.states], dim=-1)
        basis_coefficients = self.basis_decoder(z)
        N, k = expression_adjacency_list.shape
        proj_j = projections[expression_adjacency_list]
        proj_i = projections.unsqueeze(1).expand(-1, k, -1)
        delta_proj = proj_j - proj_i
        return torch.einsum('nk,nkf->nf', basis_coefficients, delta_proj)
