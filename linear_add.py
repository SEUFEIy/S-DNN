import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pprint import pprint


class DynamicLinearNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, device='cuda'):
        super(DynamicLinearNet, self).__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes.copy()
        self.device = torch.device(device=device)
        self.residual_layers = []
        self._build_network()
        self._update_residual_layers()

    def _build_network(self):
        """构建初始网络结构（含残差连接）"""
        self.layers = nn.ModuleList()
        if not self.hidden_sizes:
            self.layers.append(nn.ModuleList([nn.Linear(self.input_size, self.output_size).to(self.device)]))
            return

        # 输入层
        self.layers.append(nn.ModuleList([nn.Linear(self.input_size, self.hidden_sizes[0]).to(self.device)]))

        # 中间层（每隔两层添加残差连接）
        for i in range(1, len(self.hidden_sizes)):
            in_features = self.hidden_sizes[i-1]
            out_features = self.hidden_sizes[i]
            layer_group = nn.ModuleList([nn.Linear(in_features, out_features).to(self.device)])
            
            # 每隔两层添加残差连接
            if i % 3 == 2:
                residual = nn.Linear(self.hidden_sizes[i-2], out_features).to(self.device)
                layer_group.append(residual)
            
            self.layers.append(layer_group)

        # 输出层
        self.layers.append(nn.ModuleList([nn.Linear(self.hidden_sizes[-1], self.output_size).to(self.device)]))

    def _update_residual_layers(self):
        """更新残差层索引"""
        self.residual_layers = [i for i in range(len(self.hidden_sizes)) if (i + 1) % 3 == 0]

    def forward(self, x):
        for layer_idx, layer_group in enumerate(self.layers):
            identity = x.clone()  # 保存原始输入作为残差
            
            # 主路径处理：Conv -> BN
            x = layer_group[0](x)  # 卷积层
            x = layer_group[1](x)  # 批归一化
            
            # 残差连接处理（仅限指定层）
            if layer_idx in self.residual_layers:
                # 维度对齐：通过1x1卷积调整残差路径通道数
                residual = layer_group[2](identity)  # 残差路径卷积
                if residual.shape != x.shape:  # 自动维度匹配
                    residual = F.adaptive_avg_pool2d(residual, x.shape[2:])  # 空间维度对齐
                    residual = layer_group[3](residual)  # 通道数调整卷积
                x += residual
            
            # 激活函数 + 空间下采样
            x = self.act(x)          # 激活函数(如ReLU)
            x = self.pool(x)         # 池化压缩空间维度
            
        # 全连接层前处理
        x = self.flatten(x)          # 展平为向量
        x = self.fc(x)               # 全连接层
        return x

    def modify_neurons(self, layer_index, num_neurons):
        """修改神经元数量（兼容无残差层的场景）"""
        assert 0 <= layer_index <= len(self.layers) - 2, "Invalid layer index"
        
        current_layer = self.layers[layer_index][0]
        new_out_features = current_layer.out_features + num_neurons
        
        # 创建新的主层并复制权重
        new_main_linear = nn.Linear(current_layer.in_features, new_out_features).to(self.device)
        new_main_linear.weight.data[:current_layer.out_features] = current_layer.weight.data
        new_main_linear.bias.data[:current_layer.out_features] = current_layer.bias.data
        self.layers[layer_index][0] = new_main_linear
        
        # 更新下一层输入维度
        next_layer = self.layers[layer_index + 1][0]
        new_next_in = next_layer.in_features + num_neurons
        new_next_linear = nn.Linear(new_next_in, next_layer.out_features).to(self.device)
        new_next_linear.weight.data[:, :next_layer.in_features] = next_layer.weight.data
        new_next_linear.bias.data = next_layer.bias.data
        self.layers[layer_index + 1][0] = new_next_linear
        
        # 仅在存在残差路径时更新
        if layer_index in self.residual_layers and len(self.layers[layer_index]) > 1:
            residual_layer = self.layers[layer_index][1]
            new_residual = nn.Linear(residual_layer.in_features, new_out_features).to(self.device)
            new_residual.weight.data[:residual_layer.out_features] = residual_layer.weight.data
            new_residual.bias.data[:residual_layer.out_features] = residual_layer.bias.data
            self.layers[layer_index][1] = new_residual
        
        # 更新隐藏层参数
        self.hidden_sizes[layer_index] = new_out_features
        self._update_residual_layers()  # 关键：更新残差索引

    def add_layer(self, num_neurons):
        """添加新层（自动插入到最后一隐藏层后）"""
        last_hidden_idx = len(self.layers) - 2
        last_hidden_out = self.layers[last_hidden_idx][0].out_features
        
        # 新层和下一层
        new_layer = nn.Linear(last_hidden_out, num_neurons).to(self.device)
        new_next = nn.Linear(num_neurons, self.layers[-1][0].in_features).to(self.device)
        
        # 插入新层并更新输出层
        self.layers.insert(last_hidden_idx + 1, nn.ModuleList([new_layer]))
        self.layers[-1][0] = new_next
        
        # 更新残差索引
        self.hidden_sizes.append(num_neurons)
        self._update_residual_layers()

    def merge_sub_layers(self, layer_index):
        """
        合并指定层中的子层（兼容残差连接）
        - 如果该层包含残差路径，则合并主路径和残差路径的权重
        - 如果该层仅有主路径，按原有逻辑合并
        """
        if len(self.layers[layer_index]) == 1:
            return

        # 获取当前层的所有子层
        layer_group = self.layers[layer_index]
        main_linear = layer_group[0]
        
        # 情况1：仅有主路径（无残差）
        if len(layer_group) == 2 and not hasattr(self, 'residual_layers'):
            # 原有逻辑（合并两个正向子层）
            in_features = main_linear.in_features
            out_features = main_linear.out_features + layer_group[1].out_features
            
            merged_linear = nn.Linear(in_features, out_features).to(self.device)
            merged_linear.weight.data[:main_linear.out_features] = main_linear.weight.data
            merged_linear.weight.data[main_linear.out_features:] = layer_group[1].weight.data
            merged_linear.bias.data[:main_linear.out_features] = main_linear.bias.data
            merged_linear.bias.data[main_linear.out_features:] = layer_group[1].bias.data
            
            self.layers[layer_index] = nn.ModuleList([merged_linear])
        
        # 情况2：包含残差路径
        elif layer_index in self.residual_layers:
            # 合并主路径和残差路径的权重
            residual_linear = layer_group[1]
            merged_linear = nn.Linear(
                main_linear.in_features,
                main_linear.out_features
            ).to(self.device)
            
            # 权重融合策略（例如：求和）
            merged_linear.weight.data = main_linear.weight.data + residual_linear.weight.data
            merged_linear.bias.data = main_linear.bias.data + residual_linear.bias.data
            
            # 更新层结构（仅保留合并后的主路径）
            self.layers[layer_index] = nn.ModuleList([merged_linear])
            self._update_residual_layers()  # 更新残差索引
    

def get_features(name, features):
    def hook(modeule, input, output):
        output_np = output.detach().cpu().numpy()
        
        features[name] = output_np
        
        # if name in features:
        #     features[name] = np.concatenate((features[name], output_np))
        # else:
        #     features[name] = output_np
    
    return hook


def train(optimizer, epochs=5):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        pprint(f'Epoch [{epoch + 1}], Loss: {loss.item()}')
        net.update_neuron_survival_times()
        net.print_neuron_survial()
        net.set_dead_neuron(max_survive_times=2, dead_prop=0.5)


if __name__ == '__main__':
    input_size = 10
    output_size = 2
    hidden_sizes = [20, 30]

    net = DynamicLinearNet(input_size, output_size, hidden_sizes).cuda()

    # Dummy data
    x = torch.randn(5, input_size).cuda()
    y = torch.randint(0, output_size, (5,)).cuda()

    # Initial training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # Initial training
    print('*' * 50, 'init net', '*' * 50)
    print(net)
    train(optimizer=optimizer, epochs=5)

    # net.merge_similar_node(threshold=1)

    # ============ Dynamically add neurons =============
    # net.modify_neurons(layer_index=0, num_neurons=5)
    # print('*' * 50, 'modify neurons', '*' * 50)
    # print(net)
    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    # train(optimizer=optimizer, epochs=5)
    # net.print_neuron_survial()
    # net.merge_sub_layers(layer_index=0)
    # print('*' * 50, 'merge sub layers', '*' * 50)
    # print(net)
    # net.print_neuron_survial()
    # train(optimizer=optimizer, epochs=5)


    # net.modify_neurons(layer_index=0, num_neurons=50)
    # print('*' * 50, 'modify neurons', '*' * 50)
    # print(net)
    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    # train(optimizer=optimizer, epochs=5)
    # net.merge_sub_layers(layer_index=0)
    # print('*' * 50, 'merge sub layers', '*' * 50)
    # print(net)

    # # # ==================== Add a new layer ========================================
    # net.add_layer(layer_index=len(net.layers) - 1, num_neurons=15)
    # print('*' * 50, 'add layers', '*' * 50)
    # print(net)
    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    # train(optimizer=optimizer, epochs=5)
    # net.print_neuron_survial()

    # net.add_layer(layer_index=2, num_neurons=25)
    # print('*' * 50, 'add layers', '*' * 50)
    # print(net)
    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    # train(optimizer=optimizer, epochs=5)
