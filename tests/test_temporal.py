"""Tests for the temporal-prediction capability, the joint builder, and the vendored alignment.

All CPU-only. The train test does a single tiny epoch on a synthetic time-course.
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def _tiny_model(G, topology="full"):
    from scitoflow import VAE
    return VAE(observed=G, latent_dim=4, zr_dim=2, h_dim=2, encoder_hidden=8, decoder_hidden=8,
               t_encoder_hidden=8, graph_hidden=8, velocity_model_hidden=8, num_steps=10,
               topology=topology, use_spatial=True, use_grid_ode=True)


# ---------------------------------------------------------------- predict_at_time
def test_predict_at_time_deterministic():
    """Seeded, predict_at_time is reproducible (no hidden nondeterminism in the decode path)."""
    G, n = 16, 10
    model = _tiny_model(G)
    data = {st: torch.rand(n, G) for st in model.topo.states}
    esp = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    torch.manual_seed(0); a = model.predict_at_time(data, 0.5, edge_index_spatial=esp)["s"]
    torch.manual_seed(0); b = model.predict_at_time(data, 0.5, edge_index_spatial=esp)["s"]
    assert torch.allclose(a, b)


def test_predict_at_time_t0_collapse():
    """At target_t -> 0 every cell sits at the shared initial condition (initial_z), so the decoded
    expression is (near) identical across cells -- a clean invariant of the integrate-from-h0 design."""
    G, n = 16, 12
    model = _tiny_model(G)
    data = {st: torch.rand(n, G) for st in model.topo.states}
    esp = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    pred0 = model.predict_at_time(data, 1e-6, edge_index_spatial=esp)["s"]
    predT = model.predict_at_time(data, 1.0, edge_index_spatial=esp)["s"]
    # rows collapse at t~0 ...
    assert pred0.std(dim=0).max().item() < 1e-4
    # ... but the field actually evolves, so a later time is different
    assert (predT - pred0).abs().max().item() > 1e-6


def test_predict_at_time_per_cell_times():
    """A per-cell target-time tensor is accepted and shapes are preserved."""
    G, n = 16, 8
    model = _tiny_model(G)
    data = {st: torch.rand(n, G) for st in model.topo.states}
    esp = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    tt = torch.linspace(0.1, 0.9, n)
    pred = model.predict_at_time(data, tt, edge_index_spatial=esp)
    assert pred["s"].shape == (n, G)


# ---------------------------------------------------------------- joint builder
def test_build_joint_block_diagonal_offset():
    from scitoflow.preprocess.simulate import simulate_stage
    from scitoflow.preprocess.build_joint import build_joint
    secs = [simulate_stage(i, n_spots=36, n_genes=12, seed=0, section_id=f"E1{i}") for i in range(2)]
    joint = build_joint(secs, ["E10.0", "E12.0"], section_ids=["E10", "E12"],
                        topology="full", n_top_genes=12, use_spatial=True, offset=200.0)
    assert joint.n_obs == 72
    assert set(joint.obs["stage"]) == {"E10.0", "E12.0"}
    assert sorted(joint.obs["exp_time"].unique().tolist()) == [0, 1]
    x = joint.obs["x_position"].values
    st = joint.obs["stage"].values
    # second stage is offset well past the first (block-diagonal spatial graph)
    assert x[st == "E12.0"].min() > x[st == "E10.0"].max()


def test_build_joint_minimal_topology():
    """Spliced-only topology needs only M_s; the builder keeps it."""
    from scitoflow.preprocess.simulate import simulate_stage
    from scitoflow.preprocess.build_joint import build_joint
    secs = [simulate_stage(i, n_spots=25, n_genes=10, seed=1) for i in range(2)]
    joint = build_joint(secs, ["E10.0", "E12.0"], topology="minimal", n_top_genes=10)
    assert "M_s" in joint.layers and joint.n_obs == 50


# ---------------------------------------------------------------- alignment
def test_align_recovers_rigid_rotation():
    """A known rotation between two slices with matched features is recovered by pairwise_align."""
    import anndata as ad
    from scipy.sparse import csr_matrix
    from scitoflow.preprocess.align import pairwise_align
    rng = np.random.default_rng(0)
    n, g = 40, 24
    feats = rng.random((n, g))                       # unique per-spot features, shared by both slices
    coords = rng.random((n, 2)) * 10.0
    theta = 0.5
    Rt = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mov_coords = coords @ Rt.T + np.array([3.0, -2.0])

    def _adata(xy):
        a = ad.AnnData(X=csr_matrix(feats), layers={"M_s": csr_matrix(feats)})
        a.obs["x_position"] = xy[:, 0]; a.obs["y_position"] = xy[:, 1]
        a.var_names = [f"g{i}" for i in range(g)]
        return a

    R, t, s, method = pairwise_align(_adata(coords), _adata(mov_coords), reg=0.02, max_spots=n)
    ang = float(np.arctan2(R[1, 0], R[0, 0]))         # R maps mov -> ref, i.e. rotation by -theta
    assert abs(ang + theta) < 0.15, f"recovered angle {ang} (method {method})"


# ---------------------------------------------------------------- spatial-optional training
def test_train_vae_spatial_optional_cpu(tmp_path):
    """A 1-epoch CPU fit with the spatial factor OFF runs on the synthetic time-course."""
    from scitoflow import VAE
    from scitoflow.training.train import train_vae
    from scitoflow.preprocess.simulate import simulate_timecourse
    adata = simulate_timecourse(n_stages=3, n_spots=36, n_genes=12, seed=0, topology="full")
    model = VAE(observed=adata.n_vars, latent_dim=4, zr_dim=2, h_dim=2, encoder_hidden=8,
                decoder_hidden=8, t_encoder_hidden=8, graph_hidden=8, velocity_model_hidden=8,
                num_steps=10, topology="full", use_spatial=False, use_grid_ode=True)
    ep, ae, tj, _, _ = train_vae(
        model=model, adata=adata, epochs=1, learning_rate=1e-3, batch_size=16, grad_clip=1.0,
        test=0.2, name="t", checkpoint_folder=str(tmp_path),
        tangent_loss_params={"a": 1.0, "b": 10.0, "reg_lambda": 1.0}, device="cpu")
    assert len(ep) == 1 and np.isfinite(tj[-1])
    assert all(torch.isfinite(p).all() for p in model.parameters())
