"""
Training loop for the scIToFlow VAE (GPU-preloaded, NeighborLoader minibatches).

FAITHFUL first-pass extraction from the method-development notebook
(Base/model1new_multivelo_organized.ipynb), Phase A1 consolidation.
Logic preserved verbatim; only module imports were added/normalized.
NOT yet unit-tested or made device-agnostic (hardcoded .cuda() remains) -
that is the research-software hardening pass. See PLAN.md Phase A.
"""

from scitoflow.core.model import reindex_adjacency

# --- faithful extraction: notebook cell 20 (model1new_multivelo_organized.ipynb) ---
import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph

# Make sure the reindex_adjacency function is defined or imported here
# e.g. from .utils import reindex_adjacency

def train_vae(model, adata, epochs=50, learning_rate=1e-2, tangent_loss_params=None, batch_size=200, grad_clip=1,
              shuffle=True, test=0.1, name='', optimizer='adam', random_seed=42, checkpoint_folder=None,
              time_prior=None):
    """
    Training function optimized for GPU data handling.
    
    Key optimizations:
    1. Pre-loads data matrices (c, u, s) to GPU tensors once, avoiding
       sparse-to-dense conversion and CPU-GPU transfer in every batch.
    2. Uses global boolean masks on GPU for fast train/test splitting within batches.
    3. Removes unused model_state_history list to save memory.
    """
    
    # --- Setup Folders and Optimizer ---
    checkpoint_folder = checkpoint_folder + f'/{name}/' if checkpoint_folder is not None else './' + name + '/'
    
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)
    else:
        print('Warning, folder already exists. This may overwrite a previous fit.')
    
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, threshold=0.05, threshold_mode='rel', 
        patience=5, min_lr=1e-5,
    )
    
    # --- Optimization 1: Pre-load per-modality data to GPU (topology-driven) ---
    print("Loading data and moving to GPU...")
    states = model.topo.states
    matrices = {st: adata.layers[model.topo.layer[st]] for st in states}
    expr_matrix = matrices[model.topo.terminal]   # expression kNN on the terminal RNA state
    try:
        tensors = {st: torch.tensor(m.toarray(), dtype=torch.float64, device='cuda')
                   for st, m in matrices.items()}
        print("Successfully pre-loaded data to GPU.")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"Warning: Could not pre-load data to GPU ({e}). Using slower, on-the-fly loading.")
        tensors = None
    
    # --- Optional time prior (normalized [0,1] label per cell) preloaded to GPU ---
    time_prior_gpu = (torch.tensor(np.asarray(time_prior), dtype=torch.float64, device='cuda')
                      if time_prior is not None else None)

    # --- Graph/Adjacency setup (move to GPU) ---
    x_positions = np.vstack((adata.obs['x_position'].values, adata.obs['y_position'].values)).T
    adj_matrix_spatial = kneighbors_graph(x_positions, n_neighbors=8, mode='connectivity', include_self=False)
    edge_index_coo = adj_matrix_spatial.tocoo()
    edge_index_spatial = torch.tensor(np.vstack((edge_index_coo.row, edge_index_coo.col)), dtype=torch.long, device='cuda')
    
    # Move all static graph data to GPU
    full_data = Data(edge_index=edge_index_spatial, num_nodes=adata.n_obs).to('cuda')

    adj_matrix_expression = kneighbors_graph(expr_matrix, n_neighbors=30, mode='connectivity', include_self=False)
    # edge_index_coo_expr = adj_matrix_expression.tocoo()
    # edge_index_expression = torch.tensor(np.vstack((edge_index_coo_expr.row, edge_index_coo_expr.col)), dtype=torch.long).to('cuda')
    adj_list_expression = torch.tensor(adj_matrix_expression.nonzero()[1], dtype=torch.long, device='cuda').reshape(-1, 30)
    
    
    # --- Train/Test Split ---
    np.random.seed(random_seed)
    n_cells = adata.n_obs
    indices = np.arange(n_cells)
    if shuffle:
        np.random.shuffle(indices)
    
    n_test = int(test * n_cells)
    n_train = n_cells - n_test
    train_indices_set = set(indices[:n_train])
    test_indices_set = set(indices[n_train:])
    
    print(f"Training on {n_train} cells, testing on {n_test} cells")
    
    # --- Optimization 2: Create global train/test masks on GPU ---
    train_mask_global = torch.zeros(n_cells, dtype=torch.bool, device='cuda')
    train_mask_global[list(train_indices_set)] = True 

    test_mask_global = torch.zeros(n_cells, dtype=torch.bool, device='cuda')
    test_mask_global[list(test_indices_set)] = True
    
    # Create a tensor of test indices for the eval step
    test_indices_array = np.array(list(test_indices_set))
    test_indices_tensor = torch.tensor(test_indices_array, dtype=torch.long).cuda()
    
    # Pre-calculate the test adjacency matrix (spatial)
    test_adj_spatial = adj_matrix_spatial[test_indices_array][:, test_indices_array]
    test_edge_coo_spatial = test_adj_spatial.tocoo()
    edge_index_test_spatial = torch.tensor(np.vstack((test_edge_coo_spatial.row, test_edge_coo_spatial.col)), dtype=torch.long).cuda()

    # --- Setup Loader and Model ---
    train_loader = NeighborLoader(full_data,
                                  num_neighbors=[10, 5],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  disjoint=True)
    
    model = model.cuda()
    
    # --- Optimization 3: Removed unused model_state_history list ---
    epoch_history = [0]
    val_ae_history = [np.inf]
    val_traj_history = [np.inf]
    
    # Use a single tqdm iterator for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        train_loss_total = 0.0
        test_loss_total = 0.0
        train_batches = 0
        test_batches = 0

        # Loop directly over the data loader
        for batch in train_loader:
            optimizer.zero_grad()
            batch_idx = batch.n_id.to('cuda')  # On GPU
            batch_edge_spatial_index = batch.edge_index # On GPU
            
            # (Assuming reindex_adjacency is defined and works on-GPU)
            batch_edge_expression_index = reindex_adjacency(adj_list_expression, batch_idx, full_data.num_nodes, device='cuda')
            
            # --- Optimization 1 (Batch): per-modality data dict ---
            if tensors is not None:
                data = {st: tensors[st][batch_idx] for st in states}
            else:
                batch_idx_cpu = batch_idx.cpu().numpy()
                data = {st: torch.Tensor(matrices[st][batch_idx_cpu].toarray()).cuda() for st in states}

            loss, validation_ae, validation_traj, tangent_loss, orig_index = model.loss(
                data,
                edge_index_spatial=batch_edge_spatial_index,
                adjacency_list_expression=batch_edge_expression_index,
                tangent_loss_params=tangent_loss_params,
                epoch=epoch,
                time_prior=(time_prior_gpu[batch_idx] if time_prior_gpu is not None else None)
            )
            
            # --- Optimization 2 (Batch): Use global GPU masks ---
            # `batch_idx` and `orig_index` are both on GPU
            global_indices = batch_idx[orig_index] 
            
            # Fast, GPU-native indexing
            train_mask = train_mask_global[global_indices]
            test_mask = test_mask_global[global_indices]
            
            # --- Training Step ---
            if train_mask.sum() > 0:
                train_loss = torch.mean(loss[train_mask])
                train_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                
                train_loss_total += train_loss.item()
                train_batches += 1
                
                # Update the main progress bar with batch info
                epoch_pbar.set_postfix({
                    'Batch Train Loss': f'{train_loss.item():.3f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
                })
            
            # --- Optimization 5: Add torch.no_grad() for in-loop validation ---
            if test_mask.sum() > 0:
                with torch.no_grad():
                    test_loss = torch.mean(loss[test_mask])
                    test_loss_total += test_loss.item()
                    test_batches += 1

        avg_train_loss = train_loss_total / train_batches if train_batches > 0 else 0
        
        # --- Evaluation Step ---
        model.eval()
        with torch.no_grad():
            # --- Optimization 4: per-modality data dict for eval ---
            if tensors is not None:
                data_test = {st: tensors[st][test_indices_tensor] for st in states}
            else:
                data_test = {st: torch.Tensor(matrices[st][test_indices_array].toarray()).cuda() for st in states}

            # Use pre-computed test spatial graph
            edge_index_test = edge_index_test_spatial

            # Re-index expression graph for test set
            test_edge_expression_index = reindex_adjacency(adj_list_expression, test_indices_tensor, full_data.num_nodes, device='cuda')

            test_loss, test_validation_ae, test_validation_traj, tangent_loss, _ = model.loss(
                data_test,
                edge_index_spatial=edge_index_test,
                adjacency_list_expression=test_edge_expression_index,
                tangent_loss_params=tangent_loss_params,
                epoch=epoch,
                time_prior=(time_prior_gpu[test_indices_tensor] if time_prior_gpu is not None else None)
            )
            
            test_loss_mean = test_loss.mean().cpu().numpy()
            test_validation_ae_mean = test_validation_ae.mean().cpu().numpy()
            test_validation_traj_mean = test_validation_traj.mean().cpu().numpy()
            test_tangeng_velo_mean = tangent_loss.mean().cpu().numpy()
            
            # Use tqdm.write instead of print to avoid breaking bar
            log_msg = (f"Epoch {epoch}: Train Loss {avg_train_loss:.3f}, Test Loss {test_loss_mean:.3f}, "
                       f"Recon MSE {test_validation_ae_mean:.3f}, Traj MSE {test_validation_traj_mean:.3f}, "
                       f"Tangent Velo Loss {test_tangeng_velo_mean:.3f}")
            tqdm.write(log_msg)

        # Update LR scheduler
        scheduler_plateau.step(test_validation_traj_mean + test_validation_ae_mean)
        
        # --- Update History and Save ---
        epoch_history.append(epoch)
        val_ae_history.append(test_validation_ae_mean)
        val_traj_history.append(test_validation_traj_mean)
        # (Optimization 3: model_state_history lines removed)
        
        # Update the main progress bar with final epoch metrics
        epoch_pbar.set_postfix({
            'Train': f'{avg_train_loss:.3f}',
            'Test': f'{test_loss_mean:.3f}',
            'Recon': f'{test_validation_ae_mean:.3f}',
            'Traj': f'{test_validation_traj_mean:.3f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
        })

        torch.save(model.state_dict(), checkpoint_folder+'model_state_epoch%d.params'%(epoch))
        # (Optimization 3: del model_state_history[0] removed)
            
    # --- Load Best Model ---
    val_history = np.array(val_ae_history) + np.array(val_traj_history)
    best_index = np.argmin(val_history)

    print('Loading best model at %d epochs.'%epoch_history[best_index])
    model.load_state_dict(torch.load(
        checkpoint_folder+'model_state_epoch%d.params'%epoch_history[best_index], 
        map_location=torch.device('cuda')
    ))
    
    return np.array(epoch_history)[1:], np.array(val_ae_history)[1:], np.array(val_traj_history)[1:], edge_index_spatial, adj_list_expression

