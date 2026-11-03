"""HyperGOOD-specific spectral encoder and node energy propagation.

Clean research-release model. Scientific equivalence to the historical paper
training runs must be checked before reporting any paper-number reproduction.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .spectral import scipy_to_torch_csr


class HyperGOODNet(nn.Module):
    """Multi-frequency hypergraph encoder with structure-aware energy scores."""

    def __init__(self, input_dim, class_count, symmetric, walk, views,
                 width=64, layers=2, dropout=0.35, restart=0.5,
                 propagation_steps=2, propagation_restart=0.5):
        super().__init__()
        if not views:
            raise ValueError('At least one spectral operator is required.')
        if not 0 <= restart <= 1 or not 0 <= propagation_restart <= 1:
            raise ValueError('Restart parameters must lie in [0,1].')
        self.input_proj = nn.Linear(input_dim, width)
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.classifier = nn.Linear(width, class_count)
        self.view_logits = nn.Parameter(torch.zeros(len(views)))
        self.band_gate = nn.Parameter(torch.tensor(0.0))
        self.dropout = dropout
        self.restart = restart
        self.propagation_steps = propagation_steps
        self.propagation_restart = propagation_restart
        self.register_buffer('symmetric', scipy_to_torch_csr(symmetric))
        self.register_buffer('walk', scipy_to_torch_csr(walk))
        self.view_names = [name for name, _ in views]
        for i, (_name, matrix) in enumerate(views):
            self.register_buffer(f'operator_{i}', scipy_to_torch_csr(matrix))

    def forward(self, x):
        h0 = F.relu(self.input_proj(x))
        h = h0
        weights = F.softmax(self.view_logits, dim=0)
        for layer in self.hidden:
            channels = [torch.sparse.mm(getattr(self, f'operator_{i}'), h)
                        for i in range(len(weights))]
            mixed = torch.stack(channels).mul(weights[:, None, None]).sum(0)
            gate = torch.sigmoid(self.band_gate)
            propagated = torch.sparse.mm(self.symmetric, h)
            h = (1.0-self.restart)*(gate*mixed+(1.0-gate)*propagated) + self.restart*h0
            h = F.dropout(F.relu(layer(h)), p=self.dropout, training=self.training)
        return self.classifier(h)

    @staticmethod
    def free_energy(logits, temperature=1.0):
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        return -temperature*torch.logsumexp(logits/temperature, dim=1)

    def score(self, x, temperature=1.0, use_propagation=True):
        energy = self.free_energy(self.forward(x), temperature)
        if use_propagation:
            initial = energy
            for _ in range(self.propagation_steps):
                energy = (self.propagation_restart*initial +
                          (1.0-self.propagation_restart)*torch.sparse.mm(
                              self.walk, energy[:, None]).squeeze(1))
        return energy


def bounded_uniform_regularizer(logits, eps=1e-8):
    """Numerically safe variance penalty on logit norm and logit mass.

    This is an independently implemented optional regularizer; its exact
    coefficients require matching to the publication configuration.
    """
    lengths = torch.linalg.vector_norm(logits, dim=-1)
    mass = logits.sum(dim=-1)
    if len(logits) < 2:
        return logits.new_zeros(())
    return lengths.var(unbiased=False) / lengths.abs().mean().clamp_min(eps) + \
           0.001 * mass.var(unbiased=False) / mass.abs().mean().clamp_min(eps)
