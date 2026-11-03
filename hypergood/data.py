"""Lightweight, pickle-free hypergraph data contract for node-level experiments.

Data representation: features [N,D], labels [N], incidence arrays listing
node indices [0,N) and hyperedge indices [0,E). No PyG dependency.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class HypergraphSample:
    x: np.ndarray
    y: np.ndarray
    node_ids: np.ndarray
    edge_ids: np.ndarray
    n_edges: int

    def validate(self):
        n = self.x.shape[0]
        if self.x.ndim != 2 or n < 3 or self.x.shape[1] < 1:
            raise ValueError('x must be a [N,D] matrix with N >= 3.')
        if self.y.ndim != 1 or self.y.shape[0] != n:
            raise ValueError('y must be [N].')
        if self.node_ids.ndim != 1 or self.edge_ids.shape != self.node_ids.shape:
            raise ValueError('incidence indices must be matched 1D arrays.')
        if not np.all(np.isfinite(self.x)):
            raise ValueError('x must have finite values.')
        if self.n_edges < 1 or not len(self.node_ids):
            raise ValueError('a hypergraph must contain incidences and hyperedges.')
        if self.node_ids.min() < 0 or self.node_ids.max() >= n:
            raise ValueError('node index out of range.')
        if self.edge_ids.min() < 0 or self.edge_ids.max() >= self.n_edges:
            raise ValueError('hyperedge index out of range.')
        if len(np.unique(self.y[self.y >= 0])) < 2:
            raise ValueError('at least two labeled classes are required.')
        if not np.all(self.y == self.y.astype(np.int64)):
            raise ValueError('labels must be integer-valued.')
        return self

    def incidence(self):
        ones = np.ones(len(self.node_ids), dtype=np.float32)
        return sparse.coo_matrix((ones, (self.node_ids, self.edge_ids)),
                                 shape=(len(self.y), self.n_edges)).tocsr().sign().astype(np.float32)

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=self.x.astype('float32'), y=self.y.astype('int64'),
                            node_ids=self.node_ids.astype('int64'),
                            edge_ids=self.edge_ids.astype('int64'),
                            n_edges=np.int64(self.n_edges))


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        required = {'x', 'y', 'node_ids', 'edge_ids', 'n_edges'}
        if not required.issubset(f.files):
            raise ValueError(f'Missing required keys: {sorted(required - set(f.files))}')
        data = HypergraphSample(x=f['x'].astype(np.float32),
                                y=f['y'].astype(np.int64),
                                node_ids=f['node_ids'].astype(np.int64),
                                edge_ids=f['edge_ids'].astype(np.int64),
                                n_edges=int(f['n_edges']))
    return data.validate()


def make_toy_hypergraph(seed=7, n=90, classes=3, dimension=12):
    """Synthetic fixture only: not a benchmark and not a paper result."""
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % classes).astype(np.int64)
    rng.shuffle(y)
    centers = rng.normal(0, 1.8, (classes, dimension))
    x = centers[y] + rng.normal(0, 0.7, (n, dimension))
    node_ids, edge_ids = [], []
    e = 0
    for c in range(classes):
        candidates = np.flatnonzero(y == c)
        for _ in range(max(5, n // 6)):
            chosen = rng.choice(candidates, size=min(5, len(candidates)), replace=False)
            node_ids.extend(chosen.tolist())
            edge_ids.extend([e] * len(chosen))
            e += 1
    for _ in range(max(3, n // 20)):
        chosen = rng.choice(n, size=4, replace=False)
        node_ids.extend(chosen.tolist())
        edge_ids.extend([e] * len(chosen))
        e += 1
    return HypergraphSample(x=x.astype('float32'), y=y,
                            node_ids=np.array(node_ids, dtype=np.int64),
                            edge_ids=np.array(edge_ids, dtype=np.int64),
                            n_edges=e).validate()


def stratified_split(y, seed, train_frac=0.5, val_frac=0.25):
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError('Require train_frac > 0, val_frac > 0, sum < 1.')
    rng = np.random.default_rng(seed)
    groups = [[], [], []]
    for cls in np.unique(y[y >= 0]):
        ids = np.flatnonzero(y == cls)
        if len(ids) < 3:
            raise ValueError('Each labeled class needs at least three examples.')
        ids = rng.permutation(ids)
        n_train = max(1, int(len(ids)*train_frac))
        n_val = max(1, int(len(ids)*val_frac))
        if n_train+n_val >= len(ids):
            n_val = len(ids)-n_train-1
        for group, part in zip(groups, (ids[:n_train], ids[n_train:n_train+n_val],
                                        ids[n_train+n_val:])):
            group.extend(part.tolist())
    return tuple(np.array(sorted(ids), dtype=np.int64) for ids in groups)


def feature_shift(sample, seed, fraction=0.3, amplitude=1.0):
    """Generate a distinct test graph and explicit OOD node mask.

    IND reference scores use the same base graph and only the unaffected nodes.
    """
    if not 0.0 < fraction < 1.0 or amplitude <= 0:
        raise ValueError('fraction must be in (0,1) and amplitude positive.')
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(len(sample.y), size=max(1, int(fraction*len(sample.y))), replace=False))
    features = sample.x.copy()
    features[ids] += (rng.standard_normal((len(ids), features.shape[1])) * amplitude).astype('float32')
    shifted = HypergraphSample(features, sample.y.copy(), sample.node_ids.copy(),
                               sample.edge_ids.copy(), sample.n_edges)
    return shifted.validate(), ids
