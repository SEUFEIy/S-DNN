import warnings
warnings.filterwarnings('ignore')
import random

import os
from utils import get_color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Consolas'
plt.rcParams['legend.fontsize'] = 16

# CIFAR10 Linear
roots = [
        r'/data4/zpf/DNA-repair_S-DNN_Resnet_EWC/results/cifar100/cnn/2025-04-22===15-19-45', 
        #  r'D:\root\code\随笔\SRN_V9\results\cifar10\linear\2024-12-30===10-29-42',
        #  r'D:\root\code\随笔\SRN_V9\results\cifar10\linear\2024-12-30===10-55-47',
        #  r'D:\root\code\随笔\SRN_V9\results\cifar10\linear\2024-12-30===17-03-23',
        #  r'D:\root\code\随笔\SRN_V9\results\cifar10\linear\2024-12-30===17-03-02'
        ]

# CIFAR10 CNN
# roots = [
#         r'D:\root\code\随笔\SRN_V9\results\cifar10\cnn\2024-12-30===16-26-47', 
#          r'D:\root\code\随笔\SRN_V9\results\cifar10\cnn\2024-12-30===16-27-41',
#          r'D:\root\code\随笔\SRN_V9\results\cifar10\cnn\2024-12-30===16-50-48',
#          r'D:\root\code\随笔\SRN_V9\results\cifar10\cnn\2024-12-30===17-17-07',
#         ]
linestyle = ['--', '-.', '-', '-.', '--']
colors = get_color()
plt.figure(figsize=(8, 6))
index = 0
for root in roots:
    #effective_rank = pd.read_csv(os.path.join(root, 'rank.csv'), usecols=['effective_rank']).dropna()
    # stable_rank = pd.read_csv(os.path.join(root, 'rank.csv'), usecols=['stable_rank']).dropna()
    effective_rank = [5.91936460955245, 5.615593899721296, 5.750422535202622, 6.2668720194251994, 8.269137396148519,
                   13.138739973283755, 17.200257048415526, 14.211179191096589, 18.723563417435642, 24.497528860266755, 
                   27.290913298876816, 32.367093301998274, 26.657071056552894, 35.27104052422176, 42.447735031666156,
                   48.42342717786939, 46.611671291926314, 62.127350500652014, 61.867477222420206, 74.9673507858295]
    x_index = np.arange(len(effective_rank))
    # plt.plot(x_index, effective_rank,  marker='^', markersize=8, c=colors[index], label=f'seed {index}', linewidth=2, alpha=np.random.uniform(0.3, 0.8), linestyle=linestyle[index])
    plt.plot(x_index, effective_rank,  marker='^', markersize=8, c=colors[6], label=f'seed {index}', linewidth=2, linestyle=linestyle[index])
    index += 1

fs = 18
plt.xlabel('Task', fontsize=fs)
plt.ylabel('Effective Rank', fontsize=fs)
# plt.xticks([0, 1, 2, 3, 4], fontsize=fs)
plt.xticks(np.arange(21), fontsize=fs)
# plt.ylim([80, 100])
plt.yticks(fontsize=fs)
plt.title('Effective Rank On CIFAR100 Under CNN Model', fontsize=20)
# plt.legend()
plt.tight_layout()
plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.savefig('Effective Rank CIFAR100 CNN.png', dpi=300)
# plt.show()
