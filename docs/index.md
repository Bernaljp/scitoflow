---
sd_hide_title: true
---

# scIToFlow

<div align="center">

<h1 style="margin-bottom:0.2em;">scIToFlow</h1>

<p style="font-size:1.25em; max-width:44em; margin:0 auto 0.6em auto; color:var(--pst-color-text-muted);">
A spatial multi-omic latent neural-ODE for microenvironment-conditioned
chromatin-to-transcription dynamics.
</p>

</div>

```{image} _static/architecture.svg
:alt: scIToFlow cascade
:width: 640px
:align: center
:class: only-light
```

```{image} _static/architecture_dark.svg
:alt: scIToFlow cascade
:width: 640px
:align: center
:class: only-dark
```

scIToFlow is a variational autoencoder whose latent dynamics are a spatially conditioned
neural ordinary differential equation. It couples a chromatin channel (ATAC accessibility
or a histone modification), unspliced RNA, and spliced RNA measured at the same tissue
spots into a single generative model. The model infers a spatial latent time and evolves a
coupled cascade of regulation, chromatin, unspliced, and spliced latent states, whose drift
closes a regulatory feedback loop and is conditioned on a spatial factor learned by graph
neural networks over the tissue.

RNA velocity is one readout of this model, not its objective. The model also yields spatial
latent time, interpretable niche factors, and in-silico perturbation of either the niche or
the regulatory program.

::::{grid} 1 2 2 3
:gutter: 3
:margin: 2 0 0 0

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Install
:link: installation
:link-type: doc
Get scIToFlow with pip, including the optional preprocessing pipeline.
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Quickstart
:link: notebooks/01_quickstart
:link-type: doc
Simulate a dataset, fit the model, and read out latent time in a few minutes.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Tutorials
:link: tutorials
:link-type: doc
Executed notebooks: modality topologies, training, and spatial readouts.
:::

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Architecture
:link: architecture
:link-type: doc
The cascade equations, parameter inventory, and every loss term, laid bare.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API reference
:link: api
:link-type: doc
The public surface: `VAE`, `train_vae`, `get_topology`, `simulate_dataset`.
:::

:::{grid-item-card} {octicon}`mark-github;1.5em;sd-mr-1` Source
:link: https://github.com/Bernaljp/scitoflow
The package, preprocessing pipelines, and tests on GitHub.
:::

::::

## What the model gives you

- **Spatial latent time** — a per-spot dynamical-state coordinate, inferred jointly across
  modalities and smoothed over the tissue graph.
- **Niche factors** — a low-dimensional spatial factor `h` learned by GraphSAGE encoders
  over the spatial and expression neighbor graphs, which conditions the regulatory drift.
- **A perturbable dynamical model** — the drift closes a chromatin to transcription to
  regulation loop, so you can interrogate the fitted model with in-silico knockouts of the
  niche or the regulatory program.
- **Velocity as a readout** — tangent-space velocities on the expression graph, projectable
  to any embedding for streamline visualization.
- **Configurable modality topology** — the same model runs on the full cascade, a
  no-unspliced reduction, RNA-only variants, or a metabolic-labeling variant.

```{admonition} Honest scope
:class: important
The niche counterfactual and in-silico knockout are interrogations of the *fitted model*
(they establish a within-model dependence of the inferred regulation on the inputs), not
experimentally validated biological causation. On standard velocity-coherence metrics
scIToFlow does not outperform dedicated gene-level velocity methods; its contribution is a
spatial, generative, perturbable model, not a better velocity number.
```

```{toctree}
:hidden:
:caption: Getting started

installation
notebooks/01_quickstart
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials
notebooks/02_modality_topologies
notebooks/03_training_and_readouts
```

```{toctree}
:hidden:
:caption: Reference

architecture
api
changelog
```
