# scIToFlow

**A spatial multi-omic latent neural-ODE for microenvironment-conditioned chromatin-to-transcription dynamics.**

scIToFlow is a variational autoencoder whose latent dynamics are a spatially conditioned neural
ordinary differential equation. It couples a chromatin channel (ATAC accessibility or a histone
modification), unspliced RNA, and spliced RNA measured at the same tissue spots into a single
generative model. The model infers a spatial latent time and evolves a coupled cascade of
regulation, chromatin, unspliced, and spliced latent states, whose drift closes a regulatory
feedback loop and is conditioned on a spatial factor learned by graph neural networks over the
tissue. RNA velocity is one readout of this model, not its objective; the model also yields
spatial latent time, interpretable niche factors, and in-silico perturbation of either the niche
or the regulatory program.

**Documentation:** https://scitoflow.readthedocs.io (architecture reference, API, and
executed tutorial notebooks that run on CPU). To build the docs locally:
`pip install -e ".[docs]" && sphinx-build docs docs/_build/html`.

## Install

```bash
pip install git+https://github.com/Bernaljp/scitoflow.git
# with the moments / gene-activity preprocessing pipeline:
pip install "scitoflow[preprocess] @ git+https://github.com/Bernaljp/scitoflow.git"
```

Core dependencies are PyTorch, `torchdiffeq`, and `torch-geometric`; the `preprocess` extra adds
`scanpy` and `dynamo-release` for the moment/gene-activity build.

## Quick start

```python
import anndata as ad
from scitoflow import VAE, train_vae, get_topology

adata = ad.read_h5ad("adata_model.h5ad")  # spots x genes, with M_c/M_u/M_s moment layers + spatial coords

model = VAE(
    observed=adata.n_vars, latent_dim=16, zr_dim=2, h_dim=2,
    topology="full",          # r -> c -> u -> s -> r cascade
    use_spatial=True,         # spatial GraphSAGE niche factor h
    use_feedback=True,        # close the regulatory loop
    use_grid_ode=True,        # O(K*N) grid integration (default; ~10x cheaper than naive)
    use_expr_gnn=True,        # dual spatial + expression GNN
    tangent_reg_weight=1.0,   # tangent-space velocity regularizer
).cuda()

epochs, val_recon, val_traj, *_ = train_vae(
    model=model, adata=adata, epochs=35, batch_size=128, learning_rate=1e-3,
    tangent_loss_params={"a": 1.0, "b": 10.0, "reg_lambda": 1.0},
)
```

## Configurable modality topology

The cascade is defined over a configurable state set (`scitoflow.core.topology`), so the same
model runs on any subset of modalities:

- `full` &mdash; chromatin -> unspliced -> spliced -> regulation (the default)
- `no_unspliced` &mdash; chromatin -> spliced
- `rna_only` / `minimal` &mdash; no chromatin channel
- `labeling` &mdash; chromatin -> nascent -> total, for metabolic-labeling data

## Key ideas

- **Grid-ODE integration.** One batched trajectory is integrated over a fixed time grid and each
  cell is placed on it by differentiable linear interpolation at its latent time, which is
  equivalent to per-cell integration but linear (not quadratic) in the batch size.
- **Dual spatial conditioning.** A niche factor `h` is learned by two single-graph GraphSAGE
  encoders (spatial neighbor graph + expression neighbor graph) and conditions the drift.
- **Optional time prior.** A known time label (developmental stage or metabolic-labeling time)
  can softly supervise the latent time without forcing a per-cell correspondence.

## Data processing

The Snakemake pipelines that take each dataset from raw sequencing to the per-spot RNA and
chromatin matrices are in [`preprocessing/`](preprocessing/): spliced/unspliced RNA via kb-python and ATAC
fragments/peaks via bwa + sinto + MACS3, for spatial-Mux-seq, MISAR-seq, and
spatial-ATAC-RNA-seq. Their outputs are assembled into a model-ready AnnData by
`scitoflow.preprocess.build_dataset`. The pipelines were run on a SLURM cluster; see
[`preprocessing/README.md`](preprocessing/README.md) for accessions and how to adapt the paths.

## Honest scope

The niche counterfactual and in-silico knockout are interrogations of the *fitted model* (they
establish a within-model dependence of the inferred regulation on the inputs), not experimentally
validated biological causation. On standard velocity-coherence metrics scIToFlow does not
outperform dedicated gene-level velocity methods; its contribution is a spatial, generative,
perturbable model, not a better velocity number.

## License

MIT. See [LICENSE](LICENSE).
