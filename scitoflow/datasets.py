"""Simulated example data for tutorials, smoke tests, and quick experiments.

`simulate_dataset` produces a small, self-contained spatial multi-omic AnnData with the
layers and observations the model expects (moment-smoothed chromatin ``M_c``, unspliced
``M_u``, spliced ``M_s``; spatial coordinates ``x_position``/``y_position``). It has no
external dependencies beyond the core stack, so the documentation notebooks and the test
suite run anywhere without the (private) real datasets.

The generative process is a deliberately simple caricature of the modeled biology, not a
faithful simulator: each gene switches on along a spatially smooth latent time, and the
chromatin, unspliced, and spliced channels are lagged copies of that switch so that
chromatin leads and spliced trails. A spatial "niche" label modulates a subset of genes'
amplitudes, giving the spatial encoder a real microenvironment signal to recover. It is
meant to make the API runnable and the readouts non-trivial, not to benchmark accuracy.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import anndata as ad


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_dataset(
    n_genes: int = 60,
    grid: int = 24,
    n_niches: int = 3,
    noise: float = 0.15,
    seed: int = 0,
) -> ad.AnnData:
    """Simulate a small spatial multi-omic dataset.

    Parameters
    ----------
    n_genes : int, default 60
        Number of genes (variables).
    grid : int, default 24
        Side length of the square spot grid; the dataset has ``grid**2`` spots.
    n_niches : int, default 3
        Number of spatial niche regions. A niche multiplies the amplitude of a random
        subset of genes, so the microenvironment carries recoverable regulatory signal.
    noise : float, default 0.15
        Standard deviation of multiplicative log-normal measurement noise.
    seed : int, default 0
        Random seed; the simulation is fully determined by it.

    Returns
    -------
    anndata.AnnData
        ``adata`` with ``grid**2`` spots and ``n_genes`` genes. Layers: ``M_c`` (chromatin),
        ``M_u`` (unspliced), ``M_s`` (spliced), plus ``M_n`` (nascent) and ``M_t`` (total)
        for the metabolic-labeling topology. ``adata.X`` is a copy of ``M_s``. Observations:
        ``x_position``, ``y_position`` (grid coordinates), ``latent_time`` (the ground-truth
        time used to generate the data), ``stage`` (a coarse, noisy 0..4 developmental-stage
        label, for the time prior), and ``niche`` (the spatial region label). All count-like
        layers are stored as sparse CSR, matching the real preprocessed data.

    Examples
    --------
    >>> from scitoflow import simulate_dataset
    >>> adata = simulate_dataset(n_genes=40, grid=16, seed=0)
    >>> adata.shape
    (256, 40)
    >>> sorted(adata.layers)
    ['M_c', 'M_n', 'M_s', 'M_t', 'M_u']
    """
    rng = np.random.default_rng(seed)
    n_spots = grid * grid

    # --- spatial coordinates on a regular grid ------------------------------------
    gx, gy = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    x_pos = gx.ravel().astype(float)
    y_pos = gy.ravel().astype(float)

    # --- ground-truth latent time: a smooth diagonal gradient + mild noise --------
    t = (x_pos + y_pos) / (2.0 * (grid - 1))
    t = t + rng.normal(0.0, 0.03, size=n_spots)
    t = np.clip(t, 0.0, 1.0)

    # --- spatial niches: contiguous bands along the anti-diagonal -----------------
    band = np.clip(((x_pos + y_pos) / (2.0 * (grid - 1)) * n_niches).astype(int), 0, n_niches - 1)
    niche = band

    # --- per-gene kinetics: chromatin leads, unspliced then spliced trail ---------
    a = rng.uniform(0.1, 0.9, size=n_genes)          # activation time per gene
    w = rng.uniform(0.04, 0.15, size=n_genes)        # switch sharpness
    lag_u = rng.uniform(0.03, 0.12, size=n_genes)    # chromatin -> unspliced lag
    lag_s = rng.uniform(0.03, 0.12, size=n_genes)    # unspliced -> spliced lag
    amp = rng.uniform(1.0, 5.0, size=n_genes)        # baseline amplitude
    repressed = rng.random(n_genes) < 0.3            # some genes switch off instead of on

    # niche modulation: each niche scales a random 40% of genes by up to ~2x
    niche_gain = np.ones((n_niches, n_genes))
    for k in range(n_niches):
        hit = rng.random(n_genes) < 0.4
        niche_gain[k, hit] = rng.uniform(1.3, 2.2, size=hit.sum())

    def switch(time_vec, shift):
        z = _sigmoid((time_vec[:, None] - (a[None, :] + shift[None, :])) / w[None, :])
        return np.where(repressed[None, :], 1.0 - z, z)

    c = switch(t, np.zeros(n_genes))
    u = switch(t, lag_u)
    s = switch(t, lag_u + lag_s)

    gain = amp[None, :] * niche_gain[niche]          # (n_spots, n_genes)
    c, u, s = c * gain, u * gain, s * gain

    def measure(mat):
        mat = mat * rng.lognormal(0.0, noise, size=mat.shape)
        mat[mat < 1e-3] = 0.0                        # sparsify like real moment layers
        return sp.csr_matrix(mat.astype(np.float32))

    M_c, M_u, M_s = measure(c), measure(u), measure(s)
    M_n = M_u                                        # nascent ~ unspliced (labeling topology)
    M_t = measure(u + s)                             # total = nascent + mature

    # coarse, noisy developmental stage label for the time-prior demo
    stage = np.clip(np.round(t * 4 + rng.normal(0, 0.4, n_spots)), 0, 4).astype(int)

    adata = ad.AnnData(
        X=M_s.copy(),
        layers={"M_c": M_c, "M_u": M_u, "M_s": M_s, "M_n": M_n, "M_t": M_t},
    )
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"spot_{i}" for i in range(n_spots)]
    adata.obs["x_position"] = x_pos
    adata.obs["y_position"] = y_pos
    adata.obs["latent_time"] = t
    adata.obs["stage"] = stage
    adata.obs["niche"] = niche.astype(str)
    return adata
