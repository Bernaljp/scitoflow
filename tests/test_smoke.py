"""Smoke and CPU-path tests: the package imports, a model constructs, the example
dataset is well-formed, and a short training run completes on CPU."""
import numpy as np
import scipy.sparse as sp
import torch


def test_import():
    import scitoflow
    assert scitoflow.__version__
    assert set(["VAE", "train_vae", "get_topology", "simulate_dataset"]).issubset(scitoflow.__all__)


def test_topology_presets():
    from scitoflow import get_topology
    full = get_topology("full")
    assert full.states == ("c", "u", "s")
    assert get_topology("no_unspliced").states == ("c", "s")


def test_simulate_dataset_shape_and_layers():
    from scitoflow import simulate_dataset
    adata = simulate_dataset(n_genes=40, grid=16, seed=0)
    assert adata.shape == (256, 40)
    for layer in ("M_c", "M_u", "M_s", "M_n", "M_t"):
        assert layer in adata.layers
        assert sp.issparse(adata.layers[layer])
        assert adata.layers[layer].min() >= 0.0            # nonneg like real moment data
    for col in ("x_position", "y_position", "latent_time", "stage", "niche"):
        assert col in adata.obs


def test_simulate_dataset_is_deterministic():
    from scitoflow import simulate_dataset
    a = simulate_dataset(n_genes=20, grid=10, seed=7)
    b = simulate_dataset(n_genes=20, grid=10, seed=7)
    np.testing.assert_allclose(a.layers["M_s"].toarray(), b.layers["M_s"].toarray())


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


def test_train_vae_runs_on_cpu(tmp_path):
    """train_vae resolves the device (CPU here) and completes a short run with finite losses."""
    torch.set_default_dtype(torch.float64)
    from scitoflow import VAE, simulate_dataset, train_vae

    adata = simulate_dataset(n_genes=30, grid=14, seed=1)   # 196 spots
    model = VAE(observed=adata.n_vars, latent_dim=6, zr_dim=2, h_dim=2,
                encoder_hidden=12, decoder_hidden=12, t_encoder_hidden=12, graph_hidden=12,
                velocity_model_hidden=12, num_steps=20, ode_grid=20,
                topology="full", use_grid_ode=True, use_expr_gnn=True)

    epochs, val_recon, val_traj, edge_spatial, adj_expr = train_vae(
        model=model, adata=adata, epochs=2, batch_size=64, learning_rate=1e-2,
        tangent_loss_params={"a": 1.0, "b": 10.0, "reg_lambda": 1.0},
        checkpoint_folder=str(tmp_path),
    )
    assert len(val_recon) == 2
    assert np.all(np.isfinite(val_recon)) and np.all(np.isfinite(val_traj))

    # the fitted model yields readouts of the expected shape on CPU
    model.eval()
    data = {st: torch.tensor(adata.layers[model.topo.layer[st]].toarray(), dtype=torch.float64)
            for st in model.topo.states}
    edge_expr = model._adj_to_edge_index(adj_expr.cpu())     # (N, k) neighbor list -> [2, E]
    with torch.no_grad():
        _, latent_full, velocity, latent_time, h = model.reconstruct_latent(
            data, edge_index_spatial=edge_spatial.cpu(), edge_index_expression=edge_expr)
    assert latent_time.shape[0] == adata.n_obs
    assert velocity.shape[0] == adata.n_obs
    assert torch.isfinite(latent_time).all()
