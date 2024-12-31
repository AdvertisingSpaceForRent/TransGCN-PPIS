import torch
import torch.nn as nn
import torch.optim as optim


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # 将 logits 转化为概率
        probs = torch.sigmoid(logits)

        # 计算 BCEWithLogitsLoss without reduction
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # 提取正样本和负样本的概率
        p_t = (targets * probs) + ((1 - targets) * (1 - probs))

        # 计算调节因子 (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        # 计算平衡因子 alpha_t
        alpha_weight = (targets * self.alpha) + ((1 - targets) * (1 - self.alpha))

        # 计算最终的焦点损失
        focal_loss = alpha_weight * modulating_factor * bce_loss

        # 返回平均损失
        return focal_loss.mean()