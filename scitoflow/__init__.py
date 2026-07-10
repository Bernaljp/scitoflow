"""scIToFlow: a spatial multi-omic latent neural-ODE.

Couples chromatin, unspliced, and spliced RNA in a single latent neural ordinary
differential equation whose drift closes a regulatory feedback loop and is conditioned
on a spatial factor learned by graph neural networks over the tissue. Velocity is one
readout of the generative model, not its objective.
"""
__version__ = "0.1.0"

from scitoflow.core.model import VAE
from scitoflow.core.topology import get_topology
from scitoflow.training.train import train_vae
from scitoflow.datasets import simulate_dataset

__all__ = ["VAE", "train_vae", "get_topology", "simulate_dataset", "__version__"]
