"""Paper Eq. (14); separate from every legacy gate regularizer.

Inputs are pre-gate statistical and ADEF sequence features, with episode labels.
The denominator counts ALL ordered off-diagonal pairs, not only unlike labels.
"""
import torch
import torch.nn.functional as F


def class_conditional_decoupling(h_stat, h_clean, labels):
    if h_stat.ndim != 2 or h_stat.shape != h_clean.shape:
        raise ValueError('Expected matching [B, D] feature matrices')
    labels = labels.reshape(-1).to(h_stat.device)
    b = h_stat.shape[0]
    if labels.numel() != b:
        raise ValueError('One label is required per feature row')
    if b < 2:
        return (h_stat.sum() + h_clean.sum()) * 0.0
    cosine = F.normalize(h_stat, dim=1) @ F.normalize(h_clean, dim=1).T
    mask = labels[:, None].ne(labels[None, :])
    mask.fill_diagonal_(False)
    return (cosine.abs() * mask).sum() / (b * (b - 1))


def supervised_contrastive_loss(features, labels, temperature):
    """Exclude self-similarity from numerator AND denominator (stable SupCon)."""
    z = F.normalize(features, dim=1)
    n = len(labels)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    counts = positive.sum(1)
    valid = counts > 0
    if not valid.any():
        return features.sum()*0
    logits = (z @ z.T / temperature).masked_fill(eye, -torch.inf)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(logp.masked_fill(~positive, 0).sum(1)[valid]/counts[valid]).mean()


def factorial_loss(model, support, query, labels_query, shot, params):
    """Class-conditional loss over support+query PRE-GATE features and labels."""
    n_way = support['z'].shape[0] // shot
    proto = support['z'].reshape(n_way, shot, -1).mean(1)
    ce = F.cross_entropy(-torch.cdist(query['z'], proto).square(), labels_query)
    labels = torch.cat([torch.arange(n_way, device=labels_query.device).repeat_interleave(shot), labels_query])
    scl = supervised_contrastive_loss(torch.cat([support['z'], query['z']]), labels, params['temperature'])
    zero = query['z'].sum()*0
    dec = class_conditional_decoupling(torch.cat([support['h_stat'], query['h_stat']]),
                                     torch.cat([support['h_clean'], query['h_clean']]), labels) if model.use_dec else zero
    if model.use_gating:
        ws = torch.cat([support['w_stat'], query['w_stat']])
        wq = torch.cat([support['w_seq'], query['w_seq']])
        polar = (ws*(1-ws)+wq*(1-wq)).mean()
    else:
        polar = zero
    total = ce + params['lambda_scl']*scl + params['lambda_regularizer']*(polar+dec)
    return total, {'ce': ce, 'scl': scl, 'polar': polar, 'dec': dec, 'total': total}
