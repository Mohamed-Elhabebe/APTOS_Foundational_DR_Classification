import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        """
        Focal Loss for classification tasks.

        :param gamma: Focusing parameter (higher values down-weight easy examples)
        :param alpha: Balancing factor (useful for class imbalance)
        :param reduction: 'mean' (default) returns the average loss, 'sum' returns total loss
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute focal loss.

        :param logits: Predictions (logits) from the model, shape (batch_size, num_classes)
        :param targets: Ground truth labels, shape (batch_size,)
        """
        probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()  # One-hot encode targets

        # Compute the focal weight
        pt = (probs * targets_one_hot).sum(dim=1)  # Prob of the correct class
        focal_weight = (1 - pt) ** self.gamma

        # Compute log-probabilities
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(focal_weight * (targets_one_hot * log_probs).sum(dim=1))

        if self.alpha is not None:
            alpha_weight = self.alpha[targets] if isinstance(self.alpha, torch.Tensor) else self.alpha
            loss *= alpha_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
