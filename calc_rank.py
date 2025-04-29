import warnings
warnings.filterwarnings('ignore')
import random

import os
import torch
from utils import get_data
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import pandas as pd


def get_features(name, features):
    def hook(modeule, input, output):
        output_np = output.detach().cpu().numpy()
        
        features[name] = output_np
    
    return hook


def compute_effective_rank(singular_values: np.ndarray):
    """ Computes the effective rank of the representation layer """

    norm_sv = singular_values / np.sum(np.abs(singular_values))
    entropy = 0.0

    for p in norm_sv:
        if p > 0.0:
            entropy -= p * np.log(p)

    return np.e ** entropy


def compute_stable_rank(singular_values: np.ndarray):
    """ Computes the stable rank of the representation layer """

    sorted_singular_values = np.flip(np.sort(singular_values))
    cumsum_sorted_singular_values = np.cumsum(sorted_singular_values) / np.sum(singular_values)

    return np.sum(cumsum_sorted_singular_values < 0.99) + 1


# --------------------------------------------------------------------
def rank_linear(root, num_samples=1000):
    train_dataloader, test_dataloader, _ = get_data(data_name='cifar10', flatten=True, bs=num_samples)
    
    effective_rank = []
    stable_rank = []
    
    for task in range(5):
        print('=' * 100)
        model = torch.load(os.path.join(root, f'best_model_{task}.pth'))
        print(model)
        model = model.cuda()

        features = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d): 
                module.register_forward_hook(get_features(name, features))

        with torch.no_grad():
            for img, labels in train_dataloader:
                img, labels = img.cuda(), labels.cuda()
                _ = model(img)
                break

        feature_all = list()

        for key, _ in features.items():
            print(f'{key}: {features[key].shape}')
            feature_all.append(features[key])
    
        singular_values = svd(feature_all[-1], compute_uv=False, lapack_driver="gesvd")

        effective_rank.append(compute_effective_rank(singular_values))
        stable_rank.append(compute_stable_rank(singular_values))
    
    print(effective_rank, stable_rank)
    
    plt.figure()
    plt.subplot(121)
    plt.plot(effective_rank, marker='*', markersize=5)
    plt.xlabel('task')
    plt.title('effective rank')
    
    plt.subplot(122)
    plt.plot(stable_rank, marker='*', markersize=5)
    plt.xlabel('task')
    plt.title('stable rank')
    
    plt.tight_layout()
    # plt.savefig(os.path.join(root, 'rank.png'), dpi=300)
    
    plt.show()
    
    PlotData = {
       'effective_rank': effective_rank,
       'stable_rank': stable_rank
    }

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in PlotData.items()]))
    df.to_csv(os.path.join(root, 'rank.csv'), index=False)
    
    return effective_rank, stable_rank

    

def rank_cnn(root, num_samples=1000):
    train_dataloader, test_dataloader, _ = get_data(data_name='cifar10', flatten=False, bs=num_samples)
    
    effective_rank = []
    stable_rank = []
    
    for task in range(5):
        print('=' * 100)
        model = torch.load(os.path.join(root, f'best_model_{task}.pth'))
        print(model)
        model = model.cuda()

        features = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d): 
                module.register_forward_hook(get_features(name, features))

        with torch.no_grad():
            for img, labels in train_dataloader:
                img, labels = img.cuda(), labels.cuda()
                _ = model(img)
                break

        feature_all = list()

        for key, _ in features.items():
            print(f'{key}: {features[key].shape}')
            feature_all.append(features[key])
    
        singular_values = svd(feature_all[-1], compute_uv=False, lapack_driver="gesvd")

        effective_rank.append(compute_effective_rank(singular_values))
        stable_rank.append(compute_stable_rank(singular_values))
    
    print(effective_rank, stable_rank)
    
    plt.figure()
    plt.subplot(121)
    plt.plot(effective_rank, marker='*', markersize=5)
    plt.xlabel('task')
    plt.title('effective rank')
    
    plt.subplot(122)
    plt.plot(stable_rank, marker='*', markersize=5)
    plt.xlabel('task')
    plt.title('stable rank')
    
    plt.tight_layout()
    # plt.savefig(os.path.join(root, 'rank.png'), dpi=300)
    
    plt.show()
    PlotData = {
       'effective_rank': effective_rank,
       'stable_rank': stable_rank
    }

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in PlotData.items()]))
    df.to_csv(os.path.join(root, 'rank.csv'), index=False)
        
    return effective_rank, stable_rank


if __name__ == '__main__':
    rank_linear(root=r'results\cifar10\linear\2024-12-31===14-52-03')
    
    # rank_cnn(root=r'results\cifar10\cnn\2024-12-30===17-17-07')
    
    