"""Minimal smoke tests: the package imports and a small model constructs on CPU."""
import torch


def test_import():
    import scitoflow
    assert scitoflow.__version__
    assert set(["VAE", "train_vae", "get_topology"]).issubset(scitoflow.__all__)


def test_topology_presets():
    from scitoflow import get_topology
    full = get_topology("full")
    assert full.states == ("c", "u", "s")
    assert get_topology("no_unspliced").states == ("c", "s")


def test_vae_constructs_and_runs_on_cpu():
    torch.set_default_dtype(torch.float64)
    from scitoflow import VAE
    G = 20
    model = VAE(observed=G, latent_dim=4, zr_dim=2, h_dim=2, encoder_hidden=8, decoder_hidden=8,
                t_encoder_hidden=8, graph_hidden=8, velocity_model_hidden=8, num_steps=10,
                topology="full", use_spatial=True, use_feedback=True, use_grid_ode=True)
    assert sum(p.numel() for p in model.parameters()) > 0
    # a small forward through the latent embedding on a tiny synthetic batch
    n = 12
    data = {st: torch.rand(n, G) for st in model.topo.states}
    esp = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    out = model.latent_embedding(data, edge_index_spatial=esp)
    assert out[0].shape[0] == n
