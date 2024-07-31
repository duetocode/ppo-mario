import torch


def focal_loss(y_hat, y, alpha: torch.Tensor, gamma: float = 2):
    ce_loss = torch.nn.functional.cross_entropy(y_hat, y, reduction="none")
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return torch.mean(focal_loss)
