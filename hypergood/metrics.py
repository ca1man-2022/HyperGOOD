"""OOD metrics with explicit orientation: higher score means more OOD."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def detection_metrics(id_scores, ood_scores):
    ind = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if len(ind) == 0 or len(ood) == 0:
        raise ValueError('Detection requires nonempty ID and OOD sets.')
    scores = np.concatenate([ind, ood])
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite scores found.')
    labels = np.concatenate([np.zeros(len(ind), dtype=np.int8),
                             np.ones(len(ood), dtype=np.int8)])
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = np.flatnonzero(tpr >= 0.95)
    return {'auroc': float(roc_auc_score(labels, scores)),
            'aupr_ood_positive': float(average_precision_score(labels, scores)),
            'fpr95': float(fpr[eligible[0]]) if len(eligible) else 1.0,
            'id_count': len(ind), 'ood_count': len(ood)}
