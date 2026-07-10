# API reference

The curated public surface of scIToFlow. Everything importable directly from the top-level
`scitoflow` namespace is documented here; submodules (`preprocess`, `plotting`) expose the
data-building and plotting helpers.

## Model

```{eval-rst}
.. autoclass:: scitoflow.VAE
   :members: latent_embedding, reconstruct_latent, project_velocities, loss
   :member-order: bysource
```

## Training

```{eval-rst}
.. autofunction:: scitoflow.train_vae
```

## Topology

```{eval-rst}
.. autofunction:: scitoflow.get_topology

.. autoclass:: scitoflow.core.topology.Topology
   :members:
   :member-order: bysource
```

The available presets are `full`, `no_unspliced`, `rna_only`, `minimal`, and `labeling`
(see {doc}`notebooks/02_modality_topologies`).

## Example data

```{eval-rst}
.. autofunction:: scitoflow.simulate_dataset
```

## Preprocessing

Building a model-ready AnnData from raw matrices (requires the `preprocess` extra:
`scanpy`, `dynamo-release`).

```{eval-rst}
.. autofunction:: scitoflow.preprocess.build_dataset.build_model_adata

.. autofunction:: scitoflow.preprocess.gene_activity.calculate_gene_activity
```

## Plotting

```{eval-rst}
.. autofunction:: scitoflow.plotting.spatial.plot_obs_spatial

.. autofunction:: scitoflow.plotting.spatial.plot_history

.. autofunction:: scitoflow.plotting.velocity.plot_streamline_from_vectors
```
