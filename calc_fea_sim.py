import warnings
warnings.filterwarnings('ignore')

import os
from CKA import linear_CKA, kernel_CKA
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
        

if __name__ == '__main__':
    root = r'results\cifar10\linear\2024-12-17===14-29-24'
    model = torch.load(os.path.join(root, 'best_model.pth'))
    print(model)
    model = model.cuda()
    
    features = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d): 
            module.register_forward_hook(get_features(name, features))

    bs = 512
    train_dataloader, test_dataloader, num_classes = get_data(data_name='cifar10', flatten=True, bs=bs)
    
    with torch.no_grad():
        for img, labels in train_dataloader:
            img, labels = img.cuda(), labels.cuda()
            pred = model(img)
            break
    
    feature_all = list()

    for key, value in features.items():
        if len(value.shape) > 2:
             features[key] = np.mean(value, axis=(2, 3))
             
        print(f'{key}: {features[key].shape}')
        feature_all.append(features[key])
    
    sim_score_linearCKA = np.zeros((len(feature_all), len(feature_all)))
    sim_score_kernelCKA = np.zeros((len(feature_all), len(feature_all)))
    
    for i in range(len(feature_all)):
        for j in range(len(feature_all)):
            sim_score_linearCKA[i, j] = linear_CKA(feature_all[i], feature_all[j])
            sim_score_kernelCKA[i, j] = kernel_CKA(feature_all[i], feature_all[j])
            
    print(sim_score_linearCKA)
    print(sim_score_kernelCKA)
    
    plt.figure(figsize=(12, 6))
    fs = 12
    plt.subplot(1, 2, 1)
    plt.imshow(sim_score_linearCKA, cmap='viridis', interpolation='nearest')  # viridis  plasma  inferno  Reds  YlGnBu
    # 在每个方格中添加对应的值
    for i in range(sim_score_linearCKA.shape[0]):
        for j in range(sim_score_linearCKA.shape[1]):
            plt.text(j, i, f"{sim_score_linearCKA[i, j]:.2f}", ha="center", va="center", color="w")
    plt.title('feature  similarity  by  LinearCKA', fontsize=fs)
    plt.xlabel('Layer index')
    plt.ylabel('Layer index')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1, 2, 2)
    plt.imshow(sim_score_kernelCKA, cmap='viridis', interpolation='nearest')  # viridis  plasma  inferno  Reds  YlGnBu
    # 在每个方格中添加对应的值
    for i in range(sim_score_kernelCKA.shape[0]):
        for j in range(sim_score_kernelCKA.shape[1]):
            plt.text(j, i, f"{sim_score_kernelCKA[i, j]:.2f}", ha="center", va="center", color="w")
    plt.title('feature  similarity  by  KernelCKA', fontsize=fs)
    plt.xlabel('Layer index')
    plt.ylabel('Layer index')
    plt.xticks([])
    plt.yticks([])

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()
    plt.savefig(os.path.join(root, 'fea_sim_CKA.png'), dpi=300)
    