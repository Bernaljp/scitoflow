# Tutorials

Three executed notebooks build up from a first fit to the full spatial readout suite. Each
runs end to end on CPU using the built-in {func}`scitoflow.simulate_dataset`, so you can run
them without a GPU or any private data. Use the launch/download buttons at the top of each
notebook to open it in Colab or download the `.ipynb`.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 1 · Quickstart
:link: notebooks/01_quickstart
:link-type: doc
Simulate a spatial multi-omic dataset, fit the `full` model for a few epochs, and read out
the inferred latent time. The five-minute tour.
:::

:::{grid-item-card} 2 · Modality topologies
:link: notebooks/02_modality_topologies
:link-type: doc
The configurable state cascade: `full`, `no_unspliced`, `rna_only`, `minimal`, `labeling`.
How the encoders, decoders, and parameter counts change with the topology.
:::

:::{grid-item-card} 3 · Training and spatial readouts
:link: notebooks/03_training_and_readouts
:link-type: doc
A fuller fit, then latent time, the niche factor, the velocity streamlines, a within-model
niche counterfactual, and the optional time prior.
:::

::::

```{admonition} These notebooks use simulated data
:class: note
The datasets here come from {func}`scitoflow.simulate_dataset`, a small caricature that makes
the API runnable anywhere. It is not a benchmark: correlations and effect sizes on real
tissue will differ. For the real preprocessing pipelines see the `preprocessing/` directory
in the repository.
```
