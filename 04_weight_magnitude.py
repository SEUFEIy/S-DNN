import warnings
warnings.filterwarnings('ignore')

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import get_color


plt.rcParams['font.family'] = 'Consolas'
plt.rcParams['legend.fontsize'] = 16

# roots = [r'D:\root\code\随笔\SRN_V9\results\cifar10\linear\2024-12-30===10-26-18', 
#          r'D:\root\code\随笔\SRN_V9\results\cifar10\linear\2024-12-30===11-32-45',
#          r'D:\root\code\随笔\SRN_V9\results\cifar10\linear\2024-12-30===17-03-02',
#          r'D:\root\code\随笔\SRN_V9\results\cifar10\linear\2024-12-30===17-03-23',
#          r'D:\root\code\随笔\SRN_V9\results\cifar10\linear\2024-12-31===14-52-03']

roots = [r'/data4/zpf/DNA-repair_S-DNN_Resnet_EWC/results/cifar10/cnn/2025-04-22===22-03-46']
#roots = [r'/data4/zpf/DNA-repair_S-DNN_Resnet_EWC/results/cifar100/cnn/2025-04-22===15-19-45']
colors = get_color()
plt.figure(figsize=(8, 6))
index = 0
for root in roots:
    if 'linear' in root:
        df = pd.read_csv(os.path.join(root, 'linear_mnist.csv'), usecols=['average_weight_magnitude'])
    else:
        df = pd.read_csv(os.path.join(root, 'cnn_cifar10.csv'), usecols=['average_weight_magnitude'])
        
    df = df.dropna()
    x_index = np.arange(len(df))
    plt.plot(x_index, df,  marker='^', markersize=8, c=colors[index], label=f'seed {index}', linewidth=2, alpha=np.random.uniform(0.3, 0.8))
    index += 1
    
plt.title('Different Seed On MNIST Under Linear Model', fontsize=20)
plt.xlabel('The Invovling Sequence Tasks', fontsize=18)
plt.ylabel('Average weight magnitude', fontsize=18)
plt.xticks(np.arange(50), fontsize=12)
plt.yticks(fontsize=12)

plt.legend()
plt.tight_layout()

plt.savefig('average weight magnitude MNIST linear.png', dpi=300)
# plt.show()
