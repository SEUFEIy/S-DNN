import warnings
warnings.filterwarnings('ignore')
from ptflops import get_model_complexity_info

import torch
import os
import matplotlib.pyplot as plt
import pandas as pd

root = r'/data4/zpf/DNA-repair_S-DNN_Resnet_EWC/results/cifar10/cnn/2025-04-22===22-03-46'
#root = r'/data4/zpf/DNA-repair_S-DNN_Resnet_EWC/results/cifar100/cnn/2025-04-22===15-19-45'

model = torch.load(os.path.join(root,'best_model_4.pth'),map_location='cpu')
flops, params = get_model_complexity_info(model, input_res=(3, 32, 32), as_strings=False, print_per_layer_stat=False)
print(f'trainable_params: {params / 1e6:.2f} M, flops: {flops / 1e9:.2f} GFLOPS')