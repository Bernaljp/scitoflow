"""
Latent velocity field: a configurable coupled-ODE cascade.

Generalizes the original hardcoded zr->zc->zu->zs->zr chain to any
`scitoflow.core.topology.Topology`. One drift MLP per state node,
`f_state(parent_latent, own_latent[, t])`, plus a regulatory drift
`f_reg(terminal_latent, zr, h)`. The FULL topology reproduces the original
model exactly (same nets, same input ordering). Device-agnostic.

Latent vector layout (per row):
    [ state_0 (latent) | ... | state_{n-1} (latent) | zr (zr_dim) | h (h_dim) ]
"""
import torch
import torch.nn as nn

from scitoflow.core.networks import MLP
from scitoflow.core.topology import get_topology


class VelocityFieldReg(nn.Module):
    def __init__(self, latent, h_dim, zr_dim, hidden_dim, n_layers,
                 include_time=False, topology="full", linear_splicing=True,
                 use_feedback=True):
        super().__init__()
        self.topo = get_topology(topology)
        self.latent = latent
        self.zr_dim = zr_dim
        self.h_dim = h_dim          # already the concatenated spatial dim (2*h_dim in VAE)
        self.hidden_dim = hidden_dim
        self.include_time = include_time
        self.n_layers = n_layers
        self.n_states = self.topo.n_states
        # Ablation: when False, the regulatory node zr does not drive the first
        # cascade state (the loop zr->state_0 is cut), giving an open-loop feed-forward
        # cascade. zr is still integrated but no longer acts back on the states.
        self.use_feedback = use_feedback

        # One drift net per cascade state: input = (parent, own[, t]).
        self.state_nets = nn.ModuleDict()
        for i, st in enumerate(self.topo.states):
            parent_dim = zr_dim if i == 0 else latent
            extra = 1 if (include_time and i == 0) else 0   # time enters at the first node
            self.state_nets[st] = MLP(parent_dim + latent + extra, hidden_dim, latent,
                                      MLP_layers=n_layers)
        # Regulatory drift: (terminal_state, zr, h) -> zr
        self.reg_net = MLP(latent + zr_dim + h_dim, hidden_dim, zr_dim, MLP_layers=n_layers)

        for net in list(self.state_nets.values()) + [self.reg_net]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    nn.init.constant_(m.bias, val=0.0)

    # --- back-compat aliases so latent_reg_func can reach the splicing net ---
    @property
    def chromatin_net(self):
        return self.state_nets[self.topo.states[0]]

    @property
    def spliced_net(self):
        return self.state_nets[self.topo.production_edge[1]]

    def unspliced_net(self):
        return self.state_nets.get("u")

    def _slice(self, z):
        L, n = self.latent, self.n_states
        states = {st: z[:, i * L:(i + 1) * L] for i, st in enumerate(self.topo.states)}
        zr = z[:, n * L:n * L + self.zr_dim]
        h = z[:, n * L + self.zr_dim:]
        return states, zr, h

    def _drifts(self, t, z):
        states, zr, h = self._slice(z)
        drifts = []
        for i, st in enumerate(self.topo.states):
            own = states[st]
            if i == 0:
                parent = zr if self.use_feedback else torch.zeros_like(zr)
            else:
                parent = states[self.topo.states[i - 1]]
            if self.include_time and i == 0:
                tt = t.repeat(z.shape[0], 1) if t.dim() == 0 or t.shape[0] == 1 else t
                inp = torch.cat((parent, own, tt), dim=-1)
            else:
                inp = torch.cat((parent, own), dim=-1)
            drifts.append(self.state_nets[st](inp))
        reg_drift = self.reg_net(torch.cat((states[self.topo.terminal], zr, h), dim=-1))
        return drifts, reg_drift

    def forward(self, t, z):
        # torchdiffeq passes scalar t; keep the original broadcast behavior
        t = t.repeat(z.shape[0], 1) if t.dim() == 0 else t
        drifts, reg_drift = self._drifts(t, z)
        h_zeros = z.new_zeros(z.shape[0], self.h_dim)
        return torch.cat(drifts + [reg_drift, h_zeros], dim=-1)

    def drift(self, ts, z):
        drifts, reg_drift = self._drifts(ts, z)
        return torch.cat(drifts + [reg_drift], dim=-1)
