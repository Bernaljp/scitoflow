"""
Tiny synthetic spatial time-course generator for tests and tutorials.

`simulate_stage` makes one per-stage AnnData (a stand-in for a preprocessed `adata_full`) with the
moments layers every topology needs (M_c/M_u/M_s + raw spliced/unspliced/chromatin), a Visium-style
`array_col`/`array_row` grid, and a `var['score']` variability score. `simulate_timecourse` chains
several stages through `build_joint` to produce a joint multi-stage `adata_model` with a shared
latent-time axis -- small enough to fit the VAE for a couple of CPU epochs in the unit tests.

No dynamo / scanpy dependency: layers are simple smooth, stage-shifted, strictly-positive signals
stored as sparse matrices (the trainer expects `.toarray()`-able layers).
"""
from __future__ import annotations
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix


def simulate_stage(stage_idx=0, n_spots=64, n_genes=20, seed=0, section_id=None):
    """One synthetic stage. Expression is a smooth, strictly-positive function of a per-spot
    pseudo-time that increases with the stage index and with x, so the stages carry a shared
    developmental gradient the model can order."""
    rng = np.random.default_rng(seed + 1000 * stage_idx)
    side = int(np.ceil(np.sqrt(n_spots)))
    xx, yy = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.vstack((xx.ravel(), yy.ravel())).T[:n_spots].astype(float)
    t = stage_idx + coords[:, 0] / max(side, 1)                 # pseudo-time gradient
    freq = 0.5 + rng.random(n_genes)
    phase = rng.random(n_genes) * 2 * np.pi

    def layer(lead):
        M = np.sin((t[:, None] + lead) * freq[None, :] + phase[None, :]) + 1.3
        M = M * (1.0 + 0.08 * rng.standard_normal(M.shape))
        return csr_matrix(np.clip(M, 0.0, None))

    # chromatin leads unspliced leads spliced (a toy cascade)
    Mc, Mu, Ms = layer(0.6), layer(0.3), layer(0.0)
    sid = section_id if section_id is not None else f"sim{stage_idx}"
    n = coords.shape[0]
    import pandas as pd
    obs = pd.DataFrame({"array_col": coords[:, 0], "array_row": coords[:, 1],
                        "section": np.array([sid] * n)},
                       index=[f"{sid}_{i}" for i in range(n)])
    var = pd.DataFrame({"score": np.asarray(Ms.toarray().var(0)).ravel()},
                       index=[f"g{i}" for i in range(n_genes)])
    a = ad.AnnData(
        X=Ms.copy(), obs=obs, var=var,
        layers={"M_c": Mc, "M_u": Mu, "M_s": Ms,
                "chromatin": Mc.copy(), "unspliced": Mu.copy(), "spliced": Ms.copy()},
    )
    return a


def simulate_timecourse(n_stages=3, n_spots=64, n_genes=20, seed=0, topology="full",
                        n_top_genes=None, use_spatial=True):
    """Joint multi-stage `adata_model` (via build_joint) for tests/tutorials."""
    from scitoflow.preprocess.build_joint import build_joint
    secs = [simulate_stage(i, n_spots, n_genes, seed, section_id=f"E{10 + 2 * i}.0_sim{i}")
            for i in range(n_stages)]
    stages = [f"E{10 + 2 * i}.0" for i in range(n_stages)]
    return build_joint(secs, stages, section_ids=[s.obs["section"].iloc[0] for s in secs],
                       topology=topology, n_top_genes=(n_top_genes or n_genes),
                       gene_mode="score", use_spatial=use_spatial)
