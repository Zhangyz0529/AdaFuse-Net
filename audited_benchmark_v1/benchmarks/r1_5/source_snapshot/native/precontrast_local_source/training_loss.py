import torch
def supervised_contrastive_loss(features, labels, temperature=0.07):
    device = features.device
    similarity_matrix = torch.matmul(features, features.T) / temperature
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(features.shape[0]).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    logits = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0].detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
    valid_mask = mask.sum(1) > 0
    if valid_mask.sum() > 0:
        return -(mask * log_prob).sum(1)[valid_mask].mean()
    return torch.tensor(0.0, requires_grad=True).to(device)
