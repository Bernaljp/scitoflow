"""
Generalized joint multi-stage `adata_model` builder for a SINGLE model with a SHARED latent time.

Generalized from `scripts/build_misar_joint.py` (which was MISAR-specific). Given a set of per-stage
AnnData (each carrying the moments layers the chosen topology needs, a `var['score']` dynamo
variability score, and spatial coords), this:
  1. ranks a shared gene set across stages (the per-stage HVGs generally do not intersect),
  2. subsets every stage to that gene set,
  3. offsets each stage along x by `section_idx * offset` so the spatial kNN is BLOCK-DIAGONAL (no
     cross-stage spatial edges) while the ODE dynamics and the shared latent-time axis span all
     stages; the expression kNN stays global (links similar cell states across stages),
  4. adds an `exp_time` stage ordinal for the soft latent-time prior, and concats.

Topology-agnostic (full / rna_only / no_unspliced / minimal ...) and spatial-optional. When
`use_spatial=False` the x-offset is skipped (the spatial factor is off in training anyway).

Reused by `scripts/build_misar_joint.py` (behavior preserved) and the atlas builders
`scripts/build_mosta.py` / `scripts/build_spateo_ts.py`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import anndata as ad

from scitoflow.core.topology import get_topology

# Superset of moments/raw layers we carry through when present (topology decides which are required).
DEFAULT_LAYERS = ["spliced", "unspliced", "chromatin",
                  "M_c", "M_cc", "M_u", "M_uu", "M_s", "M_ss", "M_us", "M_n", "M_nn", "M_t", "M_tt"]


def _var_scores(sec, score_key):
    """Read just var[score_key] (backed if a path) -> Series reindexed to var_names."""
    if isinstance(sec, str):
        a = ad.read_h5ad(sec, backed="r")
    else:
        a = sec
    s = a.var[score_key] if score_key in a.var else pd.Series(np.nan, index=a.var_names)
    return s.reindex(a.var_names)


def rank_joint_genes(sections, stages, n_top_genes, gene_mode="score", score_key="score"):
    """Shared gene set across stages.

    gene_mode 'score'  : top by MEAN score across sections (dominated by early-saturating genes).
    gene_mode 'robust' : per-stage score = mean of its sections; rank = MIN across stages, so a gene
                         must be variable in EVERY stage (better developmental-ordering axis).
    """
    score_df = pd.DataFrame({i: _var_scores(sec, score_key) for i, sec in enumerate(sections)})
    if gene_mode == "robust":
        stages = np.asarray(stages)
        by_stage = {}
        for i, st in enumerate(stages):
            by_stage.setdefault(st, []).append(i)
        stage_score = pd.DataFrame({st: score_df[cols].mean(axis=1, skipna=True)
                                    for st, cols in by_stage.items()})
        rank = stage_score.min(axis=1)
    else:
        rank = score_df.mean(axis=1, skipna=True)
    return list(rank.sort_values(ascending=False).head(n_top_genes).index)


def build_joint(sections, stages, section_ids=None, topology="full", n_top_genes=100,
                gene_mode="score", use_spatial=True, offset=200.0, score_key="score"):
    """Assemble a joint multi-stage `adata_model`.

    Parameters
    ----------
    sections : list of AnnData | str
        One per stage-section (path or in-memory), each with the topology's moments layers and coords.
    stages : list
        Stage label per section (e.g. "E11.0" or a numeric age). `np.unique` sorts these into the
        `exp_time` ordinal used by the time prior.
    section_ids : list | None
        Human-readable id per section (defaults to `f"{stage}_{idx}"`); used for obs['section'] and
        as the barcode prefix so indices stay unique across stages.
    topology, n_top_genes, gene_mode, use_spatial, offset, score_key : see module docstring.
    """
    topo = get_topology(topology)
    stages = list(stages)
    assert len(sections) == len(stages), "sections and stages must be parallel"
    if section_ids is None:
        section_ids = [f"{stages[i]}_{i}" for i in range(len(sections))]

    genes = rank_joint_genes(sections, stages, n_top_genes, gene_mode, score_key)
    print(f"joint gene set ({gene_mode}): {len(genes)} genes; first 12 {genes[:12]}")

    stage_uniq = list(np.unique(np.asarray([str(s) for s in stages])))  # sorted -> ordinal
    ord_of = {s: i for i, s in enumerate(stage_uniq)}
    need = [topo.layer[st] for st in topo.states]

    parts = []
    for i, (sec, stage, sid) in enumerate(zip(sections, stages, section_ids)):
        a = ad.read_h5ad(sec) if isinstance(sec, str) else sec
        a = a[:, genes].copy()
        missing = [L for L in need if L not in a.layers]
        if missing:
            raise KeyError(f"section {sid} ({stage}) missing topology '{topo.name}' layers {missing}; "
                           f"has {list(a.layers)}")
        keep = [L for L in DEFAULT_LAYERS if L in a.layers]
        newa = ad.AnnData(X=a.X, obs=a.obs.copy(), var=a.var.loc[genes].copy(),
                          layers={L: a.layers[L].copy() for L in keep})
        newa.obs["section"] = str(sid)
        newa.obs["section_idx"] = i
        newa.obs["stage"] = str(stage)
        newa.obs["exp_time"] = ord_of[str(stage)]

        # spatial coords: prefer x/y_position, else the native Visium-style array grid.
        if "x_position" in a.obs and "y_position" in a.obs:
            x = a.obs["x_position"].values.astype(float); y = a.obs["y_position"].values.astype(float)
        elif "array_col" in a.obs and "array_row" in a.obs:
            x = a.obs["array_col"].values.astype(float); y = a.obs["array_row"].values.astype(float)
        else:
            x = y = None
        if x is not None:
            # keep the native (un-offset) grid for per-stage plotting; offset x for the graph
            newa.obs["x_position_orig"] = x
            newa.obs["y_position_orig"] = y
            if "array_col" in a.obs:
                newa.obs["array_col_orig"] = a.obs["array_col"].values
            if "array_row" in a.obs:
                newa.obs["array_row_orig"] = a.obs["array_row"].values
            newa.obs["y_position"] = y
            newa.obs["x_position"] = x + (i * offset if use_spatial else 0.0)
        newa.obs.index = [f"{sid}:{b}" for b in newa.obs.index]
        parts.append(newa)
        xr = (f"x in [{newa.obs['x_position'].min():.0f},{newa.obs['x_position'].max():.0f}]"
              if x is not None else "no coords")
        print(f"  {sid} ({stage}): {newa.shape}  {xr}")
        del a

    joint = ad.concat(parts, join="inner", merge="same", label=None)
    joint.var = parts[0].var.copy()
    print(f"DONE joint: {joint.shape} | stages {sorted(set(joint.obs['stage']))} | layers {list(joint.layers)}")
    return joint
