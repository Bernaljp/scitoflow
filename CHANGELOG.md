# Changelog

All notable changes to scIToFlow are documented here. This project follows
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- `scitoflow.simulate_dataset`: a small, self-contained spatial multi-omic example dataset
  (moment-smoothed `M_c`/`M_u`/`M_s` layers, spatial coordinates, ground-truth latent time,
  niche and stage labels) for tutorials, tests, and quick experiments.
- Documentation site (Sphinx + MyST-NB, sphinx-book-theme) with an architecture reference,
  API reference, and three executed tutorial notebooks that run on CPU.
- Snakemake data-processing pipelines under `preprocessing/` (RNA, ATAC, MISAR, Deng).

### Changed
- `scitoflow.train_vae` now resolves the compute device automatically (CUDA when available,
  otherwise CPU), so training runs unchanged on a GPU and also runs on CPU for small data.
  The GPU path is unchanged.

## [0.1.0]

### Added
- Initial public release: the spatial multi-omic latent neural-ODE `VAE`, the configurable
  modality `topology`, `train_vae`, grid-ODE integration, the dual spatial/expression
  GraphSAGE niche factor, the tangent-space velocity regularizer, and the optional soft
  time prior.
