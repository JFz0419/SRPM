"""Build ``adj_mx.pkl`` for the Seattle dataset from the released ``adj.npy``.

The raw file is expected at ``datasets/raw_data/Seattle/adj.npy`` and holds an
``N x N`` (323 x 323) node-node matrix. The released matrix is symmetric with a
zero diagonal, so the only normalisation applied here is:

  * squeeze away singleton dimensions and cast to float32,
  * symmetrise with ``max(A, A.T)`` in case of tiny numerical asymmetry,
  * zero out the diagonal (no self loop, same convention as PEMS03/04/07/08).

Weights are kept as they are released. Set ``binarize = True`` below if a purely
0/1 connectivity matrix is wanted instead.
"""
import os
import pickle

import numpy as np

dataset_name = 'Seattle'
raw_dir = f'datasets/raw_data/{dataset_name}'
adj_file_path = os.path.join(raw_dir, 'adj.npy')
output_graph_path = os.path.join(raw_dir, 'adj_mx.pkl')
binarize = False


def generate_adj_seattle():
    adj_mx = np.load(adj_file_path)
    adj_mx = np.squeeze(np.asarray(adj_mx, dtype=np.float64))

    assert adj_mx.ndim == 2, f'expect a 2-D matrix, got shape {adj_mx.shape}'
    assert adj_mx.shape[0] == adj_mx.shape[1], f'adjacency must be square, got {adj_mx.shape}'

    asymmetric = np.abs(adj_mx - adj_mx.T).max()
    if asymmetric > 0:
        adj_mx = np.maximum(adj_mx, adj_mx.T)
    np.fill_diagonal(adj_mx, 0.0)

    if binarize:
        adj_mx = (adj_mx > 0).astype(np.float32)
    else:
        adj_mx = adj_mx.astype(np.float32)

    with open(output_graph_path, 'wb') as f:
        pickle.dump(adj_mx, f)

    print(f'[adj] shape={adj_mx.shape} dtype={adj_mx.dtype} '
          f'max_asymmetry={asymmetric:.3e} nnz={(adj_mx > 0).sum()} '
          f'({(adj_mx > 0).sum() / adj_mx.size:.4%}) '
          f'value range=[{adj_mx.min():.4f}, {adj_mx.max():.4f}]')
    print(f'[adj] saved to {output_graph_path}')
    return adj_mx


if __name__ == '__main__':
    generate_adj_seattle()
