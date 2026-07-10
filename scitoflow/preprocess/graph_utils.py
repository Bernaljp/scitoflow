"""
Sparse-graph / adjacency helpers and batched function application.

FAITHFUL first-pass extraction from the method-development notebook
(Base/model1new_multivelo_organized.ipynb), Phase A1 consolidation.
Logic preserved verbatim; only module imports were added/normalized.
NOT yet unit-tested or made device-agnostic (hardcoded .cuda() remains) -
that is the research-software hardening pass. See PLAN.md Phase A.
"""

import numpy as np
import torch

# --- faithful extraction: notebook cell 5 (model1new_multivelo_organized.ipynb) ---
import os
import scipy.sparse as scp
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_scipy_sparse_matrix

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = scp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adjacency_to_edge_index(adj_matrix):
    """Convert adjacency matrix to PyTorch Geometric edge_index format"""
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
    return edge_index

def set_adj(adata):
    """Set adjacency matrix normalization"""
    adj = adata.obsp['adj']
    adj = normalize(0.9*adj + scp.sparse.eye(adj.shape[0]))
    adata.obsp['adj'] = adj

def batch_func(func, inputs, num_outputs, split_size=500):
    """Process data in batches to handle memory constraints"""
    outputs = [[] for _ in range(num_outputs)]
    
    # Determine batch size based on first tensor input
    batch_size = None
    for inp in inputs:
        if hasattr(inp, 'shape') and len(inp.shape) > 0:
            batch_size = inp.shape[0]
            break
    
    if batch_size is None:
        raise ValueError("Could not determine batch size from inputs")
    
    for i in range(0, batch_size, split_size):
        end_idx = min(i + split_size, batch_size)
        
        # Prepare batch inputs
        batch_inputs = []
        for inp in inputs:
            if inp is None:
                batch_inputs.append(None)
            elif isinstance(inp, torch.Tensor):
                batch_inputs.append(inp[i:end_idx])
            elif hasattr(inp, 'shape'):  # numpy array or similar
                batch_inputs.append(inp[i:end_idx])
            elif isinstance(inp, tuple):  # Handle batch_id tuple
                batch_tuple = []
                for sub_inp in inp:
                    if sub_inp is None:
                        batch_tuple.append(None)
                    elif hasattr(sub_inp, 'shape'):
                        batch_tuple.append(sub_inp[i:end_idx])
                    else:
                        batch_tuple.append(sub_inp)
                batch_inputs.append(tuple(batch_tuple))
            else:
                batch_inputs.append(inp)
        
        # Process batch
        batch_outputs = func(*batch_inputs)
        
        # Collect outputs
        if not isinstance(batch_outputs, (list, tuple)):
            batch_outputs = [batch_outputs]
        
        for j, output in enumerate(batch_outputs):
            if j < len(outputs):
                outputs[j].append(output)
    
    # Concatenate outputs
    final_outputs = []
    for output_list in outputs:
        if len(output_list) > 0:
            final_outputs.append(torch.cat(output_list, dim=0))
        else:
            final_outputs.append(torch.tensor([]))
    
    return tuple(final_outputs) if len(final_outputs) > 1 else final_outputs[0]

