import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss