import os
import random
import signal

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    mask = mask.bernoulli_(mask_ratio)
    return mask

def to_tensor(array):
    return torch.from_numpy(array).float()


def plot_metrics(losses, acc_list_val, roc_auc_list_val, pr_auc_list_val):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(losses) + 1)
    
    ax1.plot(epochs, losses, label='Train loss', color='#58a6ff')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, acc_list_val,     label='Accuracy', color='orange')
    ax2.plot(epochs, roc_auc_list_val, label='ROC-AUC',  color='red')
    ax2.plot(epochs, pr_auc_list_val,  label='PR-AUC',   color='green')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    
    plt.title('Training metrics')
    plt.tight_layout()
    plt.show()

class Params:
    def __init__(self):
        self.seed = 3407
        self.cuda = 0
        self.epochs = 50
        self.batch_size = 64
        self.lr = 1e-4
        self.weight_decay = 5e-2
        self.optimizer = 'AdamW'
        self.clip_value = 1
        self.dropout = 0.1
        self.classifier = 'all_patch_reps_twolayer'
        
        self.downstream_dataset = 'SHU-MI'
        self.datasets_dir = 'custom_shu/'
        self.num_of_classes = 2
        self.model_dir = 'clean_checkpoints/'
        
        self.num_workers = 16
        self.label_smoothing = 0.1
        self.multi_lr = False
        self.frozen = False
        self.use_pretrained_weights = True
        self.foundation_dir = 'pretrained_weights/pretrained_weights.pth'

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total     : {total:,}")
    print(f"Entraînable : {trainable:,}  ({100*trainable/total:.2f}%)")

def amplitude_scaling(x, scale_range=(0.8, 1.2)):
    scale = torch.FloatTensor(x.shape[0], 1, 1, 1).uniform_(*scale_range)
    return x * scale.to(x.device)

def gaussian_noise(x,  snr_db=20.0):
    signal_power = x.pow(2).mean()
    snr_linear   = 10 ** (snr_db / 10)
    noise_power  = signal_power / snr_linear
    return x + torch.randn_like(x) * noise_power.sqrt()

def augment_dataset(X, y, n_augments=3):
    X_aug = [X]
    y_aug = [y]
    
    for _ in range(n_augments):
        X_new = X.clone()
        if torch.rand(1) < 0.5:
            X_new = gaussian_noise(X_new)
        
        if torch.rand(1) < 0.5:
            X_new = amplitude_scaling(X_new)
        
        X_aug.append(X_new)
        y_aug.append(y)
    return torch.cat(X_aug, dim=0), torch.cat(y_aug, dim=0)


if __name__ == '__main__':
    a = generate_mask(192, 32, 15, mask_ratio=0.5, device=None)
    print(a)
