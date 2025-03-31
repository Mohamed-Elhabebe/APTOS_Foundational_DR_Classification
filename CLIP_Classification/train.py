import argparse
import os
import numpy as np
import random
import pandas as pd

import clip
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import *
from models import CLIPClassifier
from datasets import DRDataset
from losses import FocalLoss

parser = argparse.ArgumentParser(description='DR Classification Training')

parser.add_argument('--seed', default=777, type=int)

parser.add_argument('--model_type', default='clip', type=str)
parser.add_argument('--model_backbone', default='ViT-B/16', type=str)

parser.add_argument('--data_root_dir', default='', type=str)

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--workers', default=4, type=int)

parser.add_argument('--loss_type', default='bce', type=str)
parser.add_argument('--bce_pos_weight', default=1, type=float)
parser.add_argument('--focal_alpha', default=0.5, type=float)
parser.add_argument('--focal_gamma', default=0, type=float)

parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--wd', default=6e-5, type=float)

parser.add_argument('--num_epochs', default=100, type=int)

parser.add_argument('--result_dir', default='', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    print(f'Seed set to {args.seed}')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_type == 'clip':
        model = CLIPClassifier(args.model_backbone, device).to(device)
        _, transform = clip.load(args.model_backbone, device = device, jit = False)
    
    trainable_params, non_trainable_params = count_parameters(model)
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    def seed_worker(worker_id):
        worker_seed = args.seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset = DRDataset(os.path.join(args.data_root_dir, 'train'), transform = transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, worker_init_fn=seed_worker, generator=g, pin_memory=True, drop_last=False)

    val_dataset = DRDataset(os.path.join(args.data_root_dir, 'val'), transform = transform)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, worker_init_fn=seed_worker, generator=g, pin_memory=True, drop_last=False)
    
    if args.loss_type == 'bce':
        pos_weight = torch.tensor([args.bce_pos_weight], dtype = torch.float32, device = device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.loss_type == 'focal':
        criterion = FocalLoss(alpha = args.focal_alpha, gamma = args.focal_gamma)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas = (0.1, 0.1), eps = 1e-6, weight_decay = args.wd)

    best_f1 = 0

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    val_csv_file_path = os.path.join(args.result_dir, 'val_results.csv')
    best_checkpoint_path = os.path.join(args.result_dir, 'best_checkpoint.pth')

    for epoch in range(args.num_epochs):
        avg_loss = 0
        model.train()
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1, 1).float()

            logits = model(images)
            loss = criterion(logits, labels)
            avg_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss /= len(train_dataloader)
        print(f'Epoch {epoch}:')
        print(f'Training Loss: {avg_loss:.4f}')

        all_labels = []
        all_logits = []
        model.eval()
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images).squeeze()

                all_labels.append(labels.cpu().numpy())
                all_logits.append(logits.cpu().numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)

        accuracy, auc, f1, sensitivity, specificity = compute_metrics(all_logits, all_labels)
        print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

        log_data = {'Epoch': epoch, 'Accuracy': accuracy, 'AUC': auc, 'F1': f1, 'Sensitivity': sensitivity, 'Specificity': specificity}
        df = pd.DataFrame([log_data])
        if not os.path.isfile(val_csv_file_path):
            df.to_csv(val_csv_file_path, index = False)
        else:
            df.to_csv(val_csv_file_path, mode = 'a', header = False, index = False)
        
        # Save best checkpoint based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_checkpoint_path)
    
    print("Training Completed!")






