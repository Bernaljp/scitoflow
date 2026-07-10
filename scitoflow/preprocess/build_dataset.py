"""
Assemble the model-ready AnnData from freshly re-processed spatial-Mux-seq outputs.

FAITHFUL extraction of the notebook data-prep (Base/model1new_multivelo_organized.ipynb,
cells ~26-44): load kb-nac RNA (mature/nascent), compute ATAC gene activity, align the two
modalities by spot barcode + gene, attach spatial coordinates, run dynamo moments (with the
"chromatin-as-new-layer" trick so it is smoothed into M_c), pick HVGs, and emit `adata_model`
with the layers/obs the VAE trainer expects (M_c, M_u, M_s, x_position, y_position).

Not yet run end-to-end (waits on the alignment outputs + the `scitoflow` env's dynamo).
Run: `python -m scitoflow.preprocess.build_dataset --help`.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import dynamo as dyn

from scitoflow.preprocess.gene_activity import calculate_gene_activity


def build_model_adata(
    rna_h5ad: str,
    fragments_file: str,
    peaks_file: str,
    gtf_file: str,
    tissue_positions_csv: str,
    n_top_genes: int = 100,
    min_counts_per_cell: int = 100,
    chromatin_h5ad: str | None = None,
):
    """
    Parameters mirror the notebook. Returns (adata_full, adata_model).
      - adata_full: in-tissue spots, all HVGs, with M_c/M_u/M_s moments.
      - adata_model: HVG subset (top `n_top_genes` by dynamo score) used for training.
    """
    # 1. RNA (kb-nac): mature -> spliced, nascent -> unspliced --------------------
    adata_rna = dyn.read_h5ad(rna_h5ad)
    if "mature" in adata_rna.layers:
        adata_rna.layers["spliced"] = adata_rna.layers.pop("mature")
    if "nascent" in adata_rna.layers:
        adata_rna.layers["unspliced"] = adata_rna.layers.pop("nascent")

    # 2. Spatial coordinates (Visium-style tissue_positions_list.csv) --------------
    df_spatial = pd.read_csv(
        tissue_positions_csv, header=None,
        names=["barcode", "in_tissue", "array_row", "array_col",
               "pxl_col_in_fullres", "pxl_row_in_fullres"],
    ).set_index("barcode")
    keep = ["in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]
    adata_rna.obs[keep] = df_spatial[keep].loc[adata_rna.obs.index]
    adata_rna.obs["total_spliced"] = np.asarray(adata_rna.layers["spliced"].sum(1)).ravel()
    adata_rna.obs["total_unspliced"] = np.asarray(adata_rna.layers["unspliced"].sum(1)).ravel()
    adata_rna.obs["initial_cell_size"] = adata_rna.obs["total_spliced"] + adata_rna.obs["total_unspliced"]

    # 3. ATAC gene-activity, aligned to the RNA spots + genes ----------------------
    if chromatin_h5ad is not None:
        adata_chromatin = sc.read_h5ad(chromatin_h5ad)
    else:
        adata_chromatin = calculate_gene_activity(
            fragments_file=fragments_file, peaks_file=peaks_file, gtf_file=gtf_file,
            min_counts_per_cell=min_counts_per_cell,
        )
    obs_idx = adata_chromatin.obs.index.get_indexer_for(adata_rna.obs.index)
    var_idx = adata_chromatin.var.index.get_indexer_for(adata_rna.var.index)
    adata_chromatin = adata_chromatin[obs_idx][:, var_idx]
    adata_rna.layers["chromatin"] = adata_chromatin.X

    # 4. In-tissue spots only ------------------------------------------------------
    adata = adata_rna[adata_rna.obs["in_tissue"] == 1, :].copy()

    # 5. dynamo preprocessing + moments -------------------------------------------
    #    Stash chromatin as the 'new' layer so dyn.tl.moments smooths it into M_n,
    #    then rename the smoothed chromatin M_n -> M_c (the notebook's trick).
    adata.layers["new"] = adata.layers.pop("chromatin")
    dyn.pp.Preprocessor().preprocess_adata(adata, recipe="monocle")
    dyn.tl.moments(adata)
    adata.layers["chromatin"] = adata.layers.pop("new")
    adata.layers["X_chromatin"] = adata.layers.pop("X_new")
    adata.layers["M_c"] = adata.layers.pop("M_n")
    adata.layers["M_cc"] = adata.layers.pop("M_nn")

    # 6. Model AnnData: HVGs, spatial positions, top-N by score --------------------
    adata_model = adata[:, adata.var["use_for_pca"].values].copy()
    adata_model.obs["x_pixel"] = adata.obs["pxl_col_in_fullres"]
    adata_model.obs["y_pixel"] = adata.obs["pxl_row_in_fullres"]
    adata_model.obs["x_position"] = adata.obs["array_col"]
    adata_model.obs["y_position"] = adata.obs["array_row"]
    thresh = np.sort(adata_model.var["score"].values)[-n_top_genes]
    adata_model = adata_model[:, adata_model.var["score"].values >= thresh].copy()

    return adata, adata_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rna_h5ad", required=True, help="kb-nac counts_unfiltered/adata.h5ad")
    ap.add_argument("--fragments", required=True, help="ATAC fragments.tsv.gz (tabix-indexed)")
    ap.add_argument("--peaks", required=True, help="MACS narrowPeak")
    ap.add_argument("--gtf", required=True, help="gene annotation GTF (NCBI RefSeq GRCm38)")
    ap.add_argument("--tissue_positions", required=True, help="spatial/tissue_positions_list.csv")
    ap.add_argument("--chromatin_h5ad", default=None, help="optional precomputed gene-activity h5ad")
    ap.add_argument("--n_top_genes", type=int, default=100)
    ap.add_argument("--out", required=True, help="output adata_model .h5ad")
    args = ap.parse_args()

    _, adata_model = build_model_adata(
        rna_h5ad=args.rna_h5ad, fragments_file=args.fragments, peaks_file=args.peaks,
        gtf_file=args.gtf, tissue_positions_csv=args.tissue_positions,
        chromatin_h5ad=args.chromatin_h5ad, n_top_genes=args.n_top_genes,
    )
    adata_model.write(args.out)
    print(f"wrote {args.out}: {adata_model.shape} | layers={list(adata_model.layers)}")


if __name__ == "__main__":
    main()
