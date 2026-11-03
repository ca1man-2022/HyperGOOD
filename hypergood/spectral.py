"""Sparse hypergraph operators and explicit low/high spectral filters.

"""
import numpy as np
from scipy import sparse
import torch


def hypergraph_operators(H):
    """Return symmetric normalized propagation, random-walk propagation, L."""
    H = H.tocsr().astype(np.float64)
    d_v = np.asarray(H.sum(axis=1)).reshape(-1)
    d_e = np.asarray(H.sum(axis=0)).reshape(-1)
    if np.any(d_e <= 0):
        raise ValueError('Empty hyperedges: remove or reindex them before training.')
    inv_e = sparse.diags(np.reciprocal(d_e))
    d_sqrt = sparse.diags(np.divide(1.0, np.sqrt(d_v), out=np.zeros_like(d_v), where=d_v > 0))
    d_inv = sparse.diags(np.divide(1.0, d_v, out=np.zeros_like(d_v), where=d_v > 0))
    pair = (H @ inv_e @ H.T).tocsr()
    symmetric = (d_sqrt @ pair @ d_sqrt).tocsr()
    stochastic = (d_inv @ pair).tocsr()
    laplacian = sparse.eye(H.shape[0], format='csr') - symmetric
    return symmetric, stochastic, laplacian


def filter_family(name):
    if name == 'haar':
        return (lambda v: np.cos(v/2), lambda v: np.sin(v/2))
    if name == 'linear':
        return (lambda v: np.cos(v/2)**2, lambda v: np.sin(v)/np.sqrt(2),
                lambda v: np.sin(v/2)**2)
    if name == 'quadratic':
        return (lambda v: np.cos(v/2)**3,
                lambda v: np.sqrt(3)*np.sin(v/2)*np.cos(v/2)**2,
                lambda v: np.sqrt(3)*np.sin(v/2)**2*np.cos(v/2),
                lambda v: np.sin(v/2)**3)
    raise ValueError('Unknown filter family.')


def chebyshev_filter(L, function, order=3, scale=1.0):
    """Approximate f(scale * L) using a Chebyshev series over eigenvalues [0,2].

    This is a standard normalized-Laplacian approximation, not a numerical
    replica of earlier experimental operators.
    """
    if order < 2:
        raise ValueError('Chebyshev order must be >= 2.')
    num_nodes = L.shape[0]
    t = (np.arange(max(96, 12 * order)) + 0.5) * np.pi / max(96, 12 * order)
    y = function(scale*(1+np.cos(t)))
    coeff = np.array([(2/len(t))*np.sum(y*np.cos(k*t)) for k in range(order)])
    I = sparse.eye(num_nodes, format='csr')
    T = (L - I).tocsr()  # z = eigenvalue - 1
    prev, curr = I, T
    result = (0.5*coeff[0]*I + coeff[1]*curr).tocsr()
    for k in range(2, order):
        nxt = (2*T@curr-prev).tocsr()
        result = (result + coeff[k]*nxt).tocsr()
        prev, curr = curr, nxt
    result.eliminate_zeros()
    return result.astype('float32')


def framelet_bank(laplacian, family='haar', levels=2, order=3):
    if levels < 1:
        raise ValueError('levels must be >= 1')
    functions = filter_family(family)
    operators = []
    for scale in range(levels):
        for idx, function in enumerate(functions):
            operators.append((f'{"low" if idx == 0 else "high"}{idx}_s{scale}',
                              chebyshev_filter(laplacian, function, order=order,
                                               scale=2.0**(-scale))))
    return operators


def scipy_to_torch_csr(mat, device='cpu'):
    mat = mat.tocoo()
    indices = torch.tensor(np.stack((mat.row, mat.col)), dtype=torch.long, device=device)
    values = torch.tensor(mat.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, mat.shape, device=device).coalesce()
