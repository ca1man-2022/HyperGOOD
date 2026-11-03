"""Bounded/uniform logit regularization adapter for the public entrypoint.

"""
import torch
from torch import nn


class UB_loss(nn.Module):
    """Penalize relative variation of logit norms and summed logits.

    Signature intentionally matches the public train_OOD.py interface.
    """
    def __init__(self, lamda1=0.999, eps=1e-8):
        super().__init__()
        self.norm_weight = float(lamda1)
        self.eps = eps
        if not (0 <= self.norm_weight <= 1):
            raise ValueError('lamda1 must be between 0 and 1')

    def forward(self, _features, labels=None, _features_out=None, epoch=None):
        if _features.ndim != 2 or _features.numel() == 0:
            raise ValueError('_features must be a nonempty logits matrix')
        combined = _features if _features_out is None else torch.cat([_features, _features_out], dim=0)
        magnitudes = combined.norm(dim=1)
        avg_magnitude = magnitudes.mean().abs().detach()
        norm_penalty = magnitudes.var(unbiased=False) / avg_magnitude.clamp_min(self.eps)
        step = 0 if epoch is None else epoch
        cutoff = 120 if _features_out is not None else 20
        if step <= cutoff:
            if _features_out is None:
                return norm_penalty*0.0, avg_magnitude
            return self.norm_weight*norm_penalty, avg_magnitude

        def sum_spread(array):
            totals = array.sum(dim=1)
            scale = totals.mean().abs().detach().clamp_min(self.eps)
            return totals.var(unbiased=False) / scale

        correction = sum_spread(_features)
        if _features_out is not None:
            correction = correction + sum_spread(_features_out)
        return self.norm_weight*norm_penalty + (1.0-self.norm_weight)*correction, avg_magnitude
