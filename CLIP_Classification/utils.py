import os
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, non_trainable_params

def compute_metrics(logits, labels):
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).float().numpy()
    probs = probs.numpy()
    accuracy = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, auc, f1, sensitivity, specificity