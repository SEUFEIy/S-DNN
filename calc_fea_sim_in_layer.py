import warnings
warnings.filterwarnings('ignore')
import random

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


def plot_neuron_sim(fea_all, layer_index, num_sample, root):
    choose_index = random.sample(range(0, fea_all[layer_index].shape[1]), num_sample)

    choose_neuron_fea = []
    
    for i in choose_index:
        choose_neuron_fea.append(np.expand_dims(feature_all[layer_index][:, i], axis=0))
    
    sim_neuron_linear = np.zeros((num_sample, num_sample))
    sim_neuron_kernel = np.zeros((num_sample, num_sample))
    
    for i in range(num_sample):
        for j in range(num_sample):
            sim_neuron_linear[i, j] = linear_CKA(choose_neuron_fea[i].T, choose_neuron_fea[j].T)
            sim_neuron_kernel[i, j] = kernel_CKA(choose_neuron_fea[i].T, choose_neuron_fea[j].T)

    plt.figure(figsize=(12, 6))
    fs = 12
    
    plt.subplot(1, 2, 1)
    plt.imshow(sim_neuron_linear, cmap='viridis', interpolation='nearest')  # viridis  plasma  inferno  Reds  YlGnBu
    # 在每个方格中添加对应的值
    for i in range(sim_neuron_linear.shape[0]):
        for j in range(sim_neuron_linear.shape[1]):
            plt.text(j, i, f"{sim_neuron_linear[i, j]:.2f}", ha="center", va="center", color="w")
    plt.title(f'neuron  {layer_index + 1} similarity  by  LinearCKA', fontsize=fs)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1, 2, 2)
    plt.imshow(sim_neuron_kernel, cmap='viridis', interpolation='nearest')  # viridis  plasma  inferno  Reds  YlGnBu
    # 在每个方格中添加对应的值
    for i in range(sim_neuron_kernel.shape[0]):
        for j in range(sim_neuron_kernel.shape[1]):
            plt.text(j, i, f"{sim_neuron_kernel[i, j]:.2f}", ha="center", va="center", color="w")
    plt.title(f'neuron  {layer_index + 1} similarity  by  KernelCKA', fontsize=fs)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()
    
    # plt.show()
    plt.savefig(os.path.join(root, f'layer_{layer_index}_neuron.png'), dpi=300)


def plot_layer_sim(fea_all, root):
    sim_score_linearCKA = np.zeros((len(fea_all), len(fea_all)))
    sim_score_kernelCKA = np.zeros((len(fea_all), len(fea_all)))
    
    for i in range(len(fea_all)):
        for j in range(len(fea_all)):
            sim_score_linearCKA[i, j] = linear_CKA(fea_all[i], fea_all[j])
            sim_score_kernelCKA[i, j] = kernel_CKA(fea_all[i], fea_all[j])
            
    plt.figure(figsize=(12, 6))
    fs = 12
    plt.subplot(1, 2, 1)
    plt.imshow(sim_score_linearCKA, cmap='viridis', interpolation='nearest')  # viridis  plasma  inferno  Reds  YlGnBu
    # 在每个方格中添加对应的值
    for i in range(sim_score_linearCKA.shape[0]):
        for j in range(sim_score_linearCKA.shape[1]):
            plt.text(j, i, f"{sim_score_linearCKA[i, j]:.2f}", ha="center", va="center", color="w")
    plt.title('layer  similarity  by  LinearCKA', fontsize=fs)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1, 2, 2)
    plt.imshow(sim_score_kernelCKA, cmap='viridis', interpolation='nearest')  # viridis  plasma  inferno  Reds  YlGnBu
    # 在每个方格中添加对应的值
    for i in range(sim_score_kernelCKA.shape[0]):
        for j in range(sim_score_kernelCKA.shape[1]):
            plt.text(j, i, f"{sim_score_kernelCKA[i, j]:.2f}", ha="center", va="center", color="w")
    plt.title('layer  similarity  by  KernelCKA', fontsize=fs)
    plt.xticks([])
    plt.yticks([])

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()
    
    # plt.show()
    plt.savefig(os.path.join(root, 'fea_sim_CKA.png'), dpi=300)


if __name__ == '__main__':
    root = r'results\fashionmnist\cnn\2025-01-02===13-55-07'
    model = torch.load(os.path.join(root, 'best_model_4.pth'))
    print(model)
    model = model.cuda()
    
    features = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d): 
            module.register_forward_hook(get_features(name, features))

    bs = 512
    train_dataloader, test_dataloader, num_classes = get_data(data_name='fashionmnist', flatten=True, bs=bs)
    
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
    # ==============================================================================
    for i in range(len(feature_all)):
        plot_neuron_sim(fea_all=feature_all, layer_index=i, num_sample=10, root=root)
        
    # =========================================================================
    plot_layer_sim(fea_all=feature_all, root=root)
    