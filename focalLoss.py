import torch
import torch.nn as nn
import torch.optim as optim


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = (targets * probs) + ((1 - targets) * (1 - probs))
        modulating_factor = (1.0 - p_t) ** self.gamma
        alpha_weight = (targets * self.alpha) + ((1 - targets) * (1 - self.alpha))
        focal_loss = alpha_weight * modulating_factor * bce_loss
        return focal_loss.mean()
