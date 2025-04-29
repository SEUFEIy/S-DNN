import warnings
warnings.filterwarnings('ignore')
import random

import os
import torch
from utils import get_data
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


def get_features(name, features):
    def hook(modeule, input, output):
        output_np = output.detach().cpu().numpy()
        
        features[name] = output_np
    
    return hook


# --------------------------------------------------------------------
def calc_linear(root, task=0, num_samples=1000, dormant_unit_threshold=0.001, verbose=True):
    model = torch.load(os.path.join(root, f'best_model_{task}.pth'))
    print(model)
    model = model.cuda()

    features = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d): 
            module.register_forward_hook(get_features(name, features))

    train_dataloader, test_dataloader, _ = get_data(data_name='cifar10', flatten=True, bs=num_samples)


    for img, labels in test_dataloader:
        img, labels = img.cuda(), labels.cuda()
        pred = model(img)
        loss = nn.CrossEntropyLoss()(pred, labels)
        loss.backward()
        break

    # ================== Linear 层  =================================   
    dead_neurons = torch.zeros(len(model.layers) - 1, dtype=torch.float32)
    layer_idx = 0
    total_neurons = 0
    
    for key, value in features.items():
        if layer_idx == len(dead_neurons):
            break
        
        value = value.mean(axis=0)
        print(f'{key}: {value.shape}')
        
        total_neurons += value.shape[0]
        
        if verbose:
            plt.scatter(np.arange(len(value)), value)
            plt.show()
        
        dead_neurons[layer_idx] = (value < dormant_unit_threshold).sum()
        layer_idx += 1
        
    print(f'dead_neurons_num: {dead_neurons.sum()}, {dead_neurons.sum() / total_neurons}')
    

# ------------------------------------------------------------------
def calc_CNN(root, dormant_unit_threshold=0.001):
    model = torch.load(os.path.join(root, 'best_model_2.pth'))
    print(model)
    model = model.cuda()

    features = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d): 
            module.register_forward_hook(get_features(name, features))

    bs = 1000
    train_dataloader, test_dataloader, num_classes = get_data(data_name='cifar10', flatten=False, bs=bs)

    with torch.no_grad():
        for img, labels in train_dataloader:
            img, labels = img.cuda(), labels.cuda()
            pred = model(img)
            break

    feature_all = list()
    
    # ==================== CNN 层 ======================================     
    dead_neurons = torch.zeros(len(model.layers), dtype=torch.float32)
    layer_idx = 0
    total_neurons = 0
    
    for key, value in features.items():
        if layer_idx == len(dead_neurons):
            break
        
        if len(value.shape) > 2:
            value = np.mean(value, axis=(0, 2, 3))
        else:
            value = np.mean(value, axis=0)
            
        total_neurons += value.shape[0]
        print(f'{key}: {value.shape}')
        
        # plt.scatter(np.arange(value.shape[0]), value)
        # plt.show()
        
        dead_neurons[layer_idx] = (value < dormant_unit_threshold).sum()
        layer_idx += 1
        
    print(f'dead_neurons_num: {dead_neurons.sum()}, {dead_neurons.sum() / total_neurons}')
    
    
if __name__ == '__main__':
    # calc_linear(root=r'D:\root\code\随笔\SRN_V9\results\cifar10\linear\2024-12-31===14-52-03', dormant_unit_threshold=1e-3, task=4)
    
    # calc_CNN(root=r'results\cifar10\cnn\2024-12-30===16-50-48', dormant_unit_threshold=1e-3)
    
    root=r'results\cifar10\linear\2024-12-30===16-51-34'
    model = torch.load(os.path.join(root, f'best_model_4.pth'))
    print(model)
    
    for layer in model.layers:
        tmp = layer[0].weight.data.mean(axis=1)
        
        print((tmp < 1e-9).sum() / len(tmp))
        