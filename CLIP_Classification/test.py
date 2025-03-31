import argparse
import os
import numpy as np
import random
import pandas as pd

import clip
import torch
from torch.utils.data import DataLoader

from utils import *
from models import CLIPClassifier
from datasets import DRDataset

parser = argparse.ArgumentParser(description='DR Classification Testing')

parser.add_argument('--seed', default=777, type=int)

parser.add_argument('--model_type', default='clip', type=str)
parser.add_argument('--model_backbone', default='ViT-B/16', type=str)

parser.add_argument('--data_root_dir', default='', type=str)

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--workers', default=4, type=int)

parser.add_argument('--result_dir', default='', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    print(f'Seed set to {args.seed}')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_type == 'clip':
        model = CLIPClassifier(args.model_backbone, device).to(device)
        _, transform = clip.load(args.model_backbone, device = device, jit = False)
    
    best_checkpoint_path = os.path.join(args.result_dir, 'best_checkpoint.pth')
    model.load_state_dict(torch.load(best_checkpoint_path))
    
    def seed_worker(worker_id):
        worker_seed = args.seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    test_dataset = DRDataset(os.path.join(args.data_root_dir, 'test'), transform = transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, worker_init_fn=seed_worker, generator=g, pin_memory=True, drop_last=False)
    
    test_csv_file_path = os.path.join(args.result_dir, 'test_results.csv')

    all_labels = []
    all_logits = []
    model.eval()
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images).squeeze()

            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)

    accuracy, auc, f1, sensitivity, specificity = compute_metrics(all_logits, all_labels)
    print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

    log_data = {'Accuracy': accuracy, 'AUC': auc, 'F1': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
    df = pd.DataFrame([log_data])
    df.to_csv(test_csv_file_path, index = False)

    print("Testing Completed!")