"""
Minimal, self-contained cross-stage spatial alignment (vendored, not a dependency).

This is a compact port of the *pairwise* slice-alignment core shared by PASTE and Spateo: a fused
Gromov-Wasserstein (FGW) optimal-transport coupling between two slices' spots -- balancing feature
(expression) similarity against spatial-graph structure -- followed by a coupling-weighted rigid
Procrustes fit that maps one slice onto the other. Chaining pairwise alignments along a time-course
puts all stages into a common frame, which is what makes a *spatial* predicted-vs-measured
comparison well-posed.

We vendor only the piece we need (rather than depend on the full packages) and credit the sources:
  - PASTE : Zeira, Land, Strzalkowski & Raphael, "Alignment and integration of spatial
            transcriptomics data", Nature Methods 19, 567-575 (2022). (FGW-OT coupling +
            generalized weighted Procrustes.)
  - Spateo: Qiu et al., "Spatiotemporal modeling of molecular holograms", Cell 187 (2024).
            (`st.align`, the spatiotemporal-alignment family this follows.)

If `POT` (``import ot``) is installed we use its exact FGW solver; otherwise we fall back to an
entropic **feature-only** OT (Sinkhorn) coupling -- a documented simplification that drops the GW
structural term. The Procrustes step is identical either way.

Used by `scripts/build_mosta.py` / `scripts/build_spateo_ts.py` to store aligned coords
(`obs['x_aligned']`, `obs['y_aligned']`). Alignment feeds validation/visualization only; the joint
training graph keeps its intentional block-diagonal offset (see `build_joint`).
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import issparse
from scipy.spatial.distance import cdist


def _dense(x):
    return x.toarray() if issparse(x) else np.asarray(x)


def _features(adata, genes, layer, n_pcs, dev_seed=0):
    """Low-dim expression features on the shared genes (log1p + PCA), for the OT feature cost."""
    X = _dense(adata[:, genes].layers[layer]) if layer in adata.layers else _dense(adata[:, genes].X)
    X = np.log1p(np.clip(X, 0, None).astype(float))
    X = X - X.mean(0, keepdims=True)
    # economy SVD as PCA (no sklearn dependency here)
    k = int(min(n_pcs, X.shape[1], max(1, X.shape[0] - 1)))
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] * S[:k]


def _coords(adata, coord_keys):
    x = adata.obs[coord_keys[0]].values.astype(float)
    y = adata.obs[coord_keys[1]].values.astype(float)
    return np.vstack((x, y)).T


def _sinkhorn(M, p, q, reg, n_iter=500, tol=1e-9):
    """Entropic OT (Sinkhorn-Knopp) coupling for cost M -- the feature-only fallback."""
    K = np.exp(-M / (reg * (M.max() + 1e-12)))
    u = np.ones_like(p); v = np.ones_like(q)
    for _ in range(n_iter):
        u_prev = u
        u = p / (K @ v + 1e-300)
        v = q / (K.T @ u + 1e-300)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    return u[:, None] * K * v[None, :]


def _fgw_coupling(feat_a, feat_b, coord_a, coord_b, alpha, reg):
    """FGW coupling via POT if available, else entropic feature-only OT."""
    M = cdist(feat_a, feat_b, metric="sqeuclidean")
    M = M / (M.max() + 1e-12)
    p = np.ones(feat_a.shape[0]) / feat_a.shape[0]
    q = np.ones(feat_b.shape[0]) / feat_b.shape[0]
    try:
        import ot  # POT
        Da = cdist(coord_a, coord_a); Db = cdist(coord_b, coord_b)
        Da = Da / (Da.max() + 1e-12); Db = Db / (Db.max() + 1e-12)
        pi = ot.gromov.fused_gromov_wasserstein(M, Da, Db, p, q, loss_fun="square_loss", alpha=alpha)
        return np.asarray(pi), "fgw(POT)"
    except Exception:
        return _sinkhorn(M, p, q, reg), "sinkhorn(feature-only)"


def weighted_procrustes(ref_xy, mov_xy, pi, allow_scale=False):
    """Coupling-weighted rigid Procrustes mapping `mov_xy` onto `ref_xy` (PASTE-style).

    Returns (R, t, s) s.t. aligned = s * mov_xy @ R.T + t best matches ref under coupling pi.
    """
    wr = pi.sum(1); wm = pi.sum(0)
    mu_r = (wr[:, None] * ref_xy).sum(0) / (wr.sum() + 1e-12)
    mu_m = (wm[:, None] * mov_xy).sum(0) / (wm.sum() + 1e-12)
    Rc = ref_xy - mu_r; Mc = mov_xy - mu_m
    # (2,2) cross-covariance ref<-mov under the coupling: sum_ij pi_ij Rc_i Mc_j^T = Rc.T @ pi @ Mc
    C = Rc.T @ pi @ Mc
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(2); D[1, 1] = np.sign(np.linalg.det(U @ Vt))   # reflection fix -> proper rotation
    R = U @ D @ Vt
    s = 1.0
    if allow_scale:
        var_m = (wm * (Mc ** 2).sum(1)).sum() / (wm.sum() + 1e-12)
        s = float((S * np.diag(D)).sum() / (var_m + 1e-12))
    t = mu_r - s * (R @ mu_m)
    return R, t, s


def _apply(xy, R, t, s):
    return s * (xy @ R.T) + t


def pairwise_align(ref, mov, genes=None, layer="M_s", coord_keys=("x_position", "y_position"),
                   alpha=0.1, reg=0.05, n_pcs=30, max_spots=2000, allow_scale=False, seed=0):
    """Estimate the rigid transform aligning `mov` onto `ref`. Returns (R, t, s, method).

    The OT coupling is estimated on a subsample of <= `max_spots` spots per slice (FGW/Sinkhorn are
    O(n^2) in spots); the resulting rigid transform is then applied to *all* of `mov`'s spots.
    """
    if genes is None:
        genes = [g for g in ref.var_names if g in set(mov.var_names)]
    if len(genes) == 0:
        raise ValueError("no shared genes between the two slices to align on")
    rng = np.random.default_rng(seed)

    def _sub(a):
        idx = np.arange(a.n_obs)
        if a.n_obs > max_spots:
            idx = rng.choice(idx, max_spots, replace=False)
        return idx

    ir, im = _sub(ref), _sub(mov)
    fa = _features(ref[ir], genes, layer, n_pcs)
    fb = _features(mov[im], genes, layer, n_pcs)
    ca = _coords(ref[ir], coord_keys); cb = _coords(mov[im], coord_keys)
    pi, method = _fgw_coupling(fa, fb, ca, cb, alpha, reg)
    R, t, s = weighted_procrustes(ca, cb, pi, allow_scale=allow_scale)
    return R, t, s, method


def align_sections(adatas, ref_index=0, coord_keys=("x_position", "y_position"), layer="M_s",
                   out_keys=("x_aligned", "y_aligned"), **kw):
    """Align a list of per-stage AnnData into a common frame by chaining pairwise alignments.

    Each stage is aligned onto the (already-aligned) previous stage, walking outward from
    `ref_index`. Writes aligned coords into `obs[out_keys]` in place and returns the list.
    """
    n = len(adatas)
    for a in adatas:
        a.obs[out_keys[0]] = _coords(a, coord_keys)[:, 0]
        a.obs[out_keys[1]] = _coords(a, coord_keys)[:, 1]
    order = list(range(ref_index + 1, n)) + list(range(ref_index - 1, -1, -1))
    for i in order:
        ref = adatas[i + 1] if i < ref_index else adatas[i - 1]
        R, t, s, method = pairwise_align(ref, adatas[i], coord_keys=out_keys, layer=layer, **kw)
        xy = _coords(adatas[i], out_keys)
        aligned = _apply(xy, R, t, s)
        adatas[i].obs[out_keys[0]] = aligned[:, 0]
        adatas[i].obs[out_keys[1]] = aligned[:, 1]
        print(f"  aligned stage {i} -> ref {(i + 1) if i < ref_index else (i - 1)} via {method}")
    return adatas
