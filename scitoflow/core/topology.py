"""
Configurable modality topology for the scIToFlow latent cascade.

The model integrates an ordered chain of latent state nodes closed by a regulatory
node `r` and conditioned on a spatial variable `h`:

    r -> state[0] -> state[1] -> ... -> state[-1] -> r   (feedback), r also sees h

Each state node has a latent (`latent_dim`) representation; its drift depends on its
parent (the previous state, or `r` for the first state). The regulatory drift depends
on the terminal state, `r`, and `h`. A subset of states (`decoded`) is reconstructed to
gene space and carries a data likelihood (the RNA-like readouts); the others (e.g.
chromatin) are encoded drivers only.

Presets
-------
- FULL          r->c->u->s->r        chromatin + unspliced + spliced (default; the
                                     original scIToFlow model)
- NO_UNSPLICED  r->c->s->r           chromatin + spliced (lumps transcription+splicing
                                     into one chromatin->spliced production step when
                                     unspliced is not measured)
- LABELING      r->c->nascent->total metabolic labeling (new vs total RNA); `total` is
                                     reconstructed as nascent + mature
- RNA_ONLY      r->u->s->r           no chromatin
- MINIMAL       r->s->r              spliced only

`layer` maps each state to the AnnData moments layer that feeds its encoder.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Topology:
    name: str
    states: tuple           # ordered cascade of latent state names, e.g. ("c","u","s")
    decoded: tuple          # states reconstructed to gene space (RNA-like readouts)
    layer: dict             # state name -> AnnData layer holding its (moments) data

    @property
    def n_states(self) -> int:
        return len(self.states)

    def parent(self, state: str):
        """Drift parent of a state: 'r' for the first, else the previous state."""
        i = self.states.index(state)
        return "r" if i == 0 else self.states[i - 1]

    @property
    def terminal(self) -> str:
        """The state that feeds the regulatory node (last in the chain)."""
        return self.states[-1]

    @property
    def production_edge(self):
        """(parent, child) of the last cascade edge -- where the positive-rate
        biophysical regularizer applies (splicing in FULL, chromatin->spliced in
        NO_UNSPLICED)."""
        return (self.parent(self.terminal), self.terminal)

    def index(self, state: str) -> int:
        return self.states.index(state)


PRESETS = {
    "full":         Topology("full",        ("c", "u", "s"),             ("u", "s"),
                             {"c": "M_c", "u": "M_u", "s": "M_s"}),
    "no_unspliced": Topology("no_unspliced", ("c", "s"),                 ("s",),
                             {"c": "M_c", "s": "M_s"}),
    "rna_only":     Topology("rna_only",     ("u", "s"),                 ("u", "s"),
                             {"u": "M_u", "s": "M_s"}),
    "minimal":      Topology("minimal",      ("s",),                     ("s",),
                             {"s": "M_s"}),
    "labeling":     Topology("labeling",     ("c", "nascent", "total"),  ("nascent", "total"),
                             {"c": "M_c", "nascent": "M_n", "total": "M_t"}),
}


def get_topology(spec) -> Topology:
    """Resolve a preset name or pass through a Topology."""
    if isinstance(spec, Topology):
        return spec
    key = str(spec).lower()
    if key not in PRESETS:
        raise ValueError(f"unknown topology '{spec}'; choose from {list(PRESETS)} or pass a Topology")
    return PRESETS[key]
