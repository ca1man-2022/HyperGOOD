#!/usr/bin/env python3
"""Standalone reproducibility scaffold for the HyperGOOD spectral-energy pipeline.
"""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from hypergood.data import load_npz, stratified_split, feature_shift
from hypergood.metrics import detection_metrics
from hypergood.model import HyperGOODNet, bounded_uniform_regularizer
from hypergood.spectral import hypergraph_operators, framelet_bank


def parser():
    p = argparse.ArgumentParser(description='Train/evaluate an independent HyperGOOD model')
    p.add_argument('--dataset', type=Path, required=True, help='Standardized COO incidence NPZ')
    p.add_argument('--output', type=Path, default=Path('outputs/hypergood_run.json'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu', help='cpu, cuda, cuda:0, ...')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.35)
    p.add_argument('--learning-rate', type=float, default=0.002)
    p.add_argument('--weight-decay', type=float, default=0.001)
    p.add_argument('--train-fraction', type=float, default=0.5)
    p.add_argument('--validation-fraction', type=float, default=0.25)
    p.add_argument('--filter-family', choices=['haar', 'linear', 'quadratic'], default='haar')
    p.add_argument('--levels', type=int, default=2)
    p.add_argument('--chebyshev-order', type=int, default=3)
    p.add_argument('--restart', type=float, default=0.5)
    p.add_argument('--propagation-steps', type=int, default=2)
    p.add_argument('--propagation-restart', type=float, default=0.5)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--without-energy-propagation', action='store_true')
    p.add_argument('--regularization-weight', type=float, default=0.0)
    p.add_argument('--ood-fraction', type=float, default=0.3)
    p.add_argument('--ood-noise', type=float, default=1.0)
    p.add_argument('--verbose', action='store_true')
    return p


def run(args):
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    original = load_npz(args.dataset)
    shifted, ood_indices = feature_shift(original, seed=args.seed+8191,
                                         fraction=args.ood_fraction,
                                         amplitude=args.ood_noise)
    id_mask = np.ones(len(original.y), dtype=bool)
    id_mask[ood_indices] = False
    id_indices = np.flatnonzero(id_mask)
    if not len(id_indices):
        raise ValueError('No untouched IND nodes remain.')

    # Holdout is determined exclusively from ID labels; OOD masks never
    # influence checkpoint/model selection or learning rate schedules.
    train_indices, val_indices, test_indices = stratified_split(
        original.y, args.seed, args.train_fraction, args.validation_fraction)
    original_y = original.y.copy()
    classes = np.unique(original_y[original_y >= 0])
    remapped = np.full(len(original_y), -1, dtype=np.int64)
    for i, label in enumerate(classes):
        remapped[original_y == label] = i

    symmetric, walk, laplacian = hypergraph_operators(original.incidence())
    operators = framelet_bank(laplacian, family=args.filter_family,
                              levels=args.levels, order=args.chebyshev_order)
    model = HyperGOODNet(original.x.shape[1], len(classes), symmetric,
                         walk, operators, width=args.width, layers=args.layers,
                         dropout=args.dropout, restart=args.restart,
                         propagation_steps=args.propagation_steps,
                         propagation_restart=args.propagation_restart).to(device)
    x_id = torch.tensor(original.x, dtype=torch.float32, device=device)
    x_ood = torch.tensor(shifted.x, dtype=torch.float32, device=device)
    y = torch.tensor(remapped, dtype=torch.long, device=device)
    tr_idx = torch.tensor(train_indices, dtype=torch.long, device=device)
    va_idx = torch.tensor(val_indices, dtype=torch.long, device=device)
    te_idx = torch.tensor(test_indices, dtype=torch.long, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                             weight_decay=args.weight_decay)
    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        logits = model(x_id)
        loss = F.cross_entropy(logits[tr_idx], y[tr_idx])
        if args.regularization_weight:
            loss = loss + args.regularization_weight*bounded_uniform_regularizer(logits[tr_idx])
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at epoch {epoch+1}')
        loss.backward()
        optim.step()
        model.eval()
        with torch.no_grad():
            output = model(x_id)
            va_acc = (output[va_idx].argmax(1) == y[va_idx]).float().mean().item()
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_epoch = epoch + 1
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
        if args.verbose and ((epoch+1) % 25 == 0 or epoch+1 == args.epochs):
            print(f'Epoch {epoch+1}: loss={loss.item():.4f} val_acc={va_acc:.4f}')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x_id)
        test_accuracy = (logits[te_idx].argmax(1) == y[te_idx]).float().mean().item()
        score_ind = model.score(x_id, temperature=args.temperature,
                                use_propagation=not args.without_energy_propagation)
        score_ood = model.score(x_ood, temperature=args.temperature,
                                use_propagation=not args.without_energy_propagation)
        # Evaluate only unchanged ID node-context pairs and explicitly altered
        # OOD node-context pairs. Do not score entire graph as OOD.
        metrics = detection_metrics(score_ind[id_indices].cpu().numpy(),
                                    score_ood[ood_indices].cpu().numpy())

    report = {'status': 'SCAFFOLD_NOT_PAPER_REPRODUCTION',
              'dataset': str(args.dataset), 'seed': args.seed, 'device': str(device),
              'configuration': vars(args).copy(),
              'selection': {'criterion': 'id_validation_accuracy',
                            'best_epoch': best_epoch, 'best_id_validation_accuracy': best_val_acc},
              'id_test_accuracy': test_accuracy, 'detection': metrics,
              'ood_protocol': {'type': 'feature', 'fraction': args.ood_fraction,
                               'noise_amplitude': args.ood_noise,
                               'shift_seed': args.seed+8191,
                               'reference': 'unchanged base graph',
                               'positive': 'perturbed nodes in shifted graph',
                               'negative': 'untouched nodes in reference graph'},
              'limitations': [
                  'Independent reconstruction; not yet numerically matched to published results.',
                  'Synthetic or user-supplied data only: real benchmark data not bundled.',
                  'No guarantee of literal match with historical training protocol.']}
    report['configuration']['dataset'] = str(args.dataset)
    report['configuration']['output'] = str(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    print(json.dumps({'output': str(args.output), 'id_test_acc': round(test_accuracy,4),
                      'detection': metrics, 'best_epoch': best_epoch}, indent=2))
    return report


if __name__ == '__main__':
    run(parser().parse_args())
