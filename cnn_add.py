import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DynamicCNN(nn.Module):
    def __init__(self, input_channels, conv_hidden_layers, num_classes, size=32, device='cuda'):
        super(DynamicCNN, self).__init__()
        self.layers = nn.ModuleList()
        self.num_class = num_classes
        self.device = torch.device(device=device)
        self.input_channels = input_channels
        self.hidden_sizes = conv_hidden_layers.copy()  # 关键修复：保存为实例属性
        self.size = size
        self.ewc_params = {}       # 存储各任务最优参数 {task_id: {param_name: tensor}}
        self.fisher = {}           # 存储各任务Fisher信息 {task_id: {param_name: tensor}}
        self.plasticity_mask = {}  # 可塑性掩码 {param_name: mask}
        self.task_count = 0        # 已学习任务计数器
        
        # 自动生成每隔两层的残差层索引（如第2、5、8层）
        self.residual_layers = [i for i in range(len(self.hidden_sizes)) if (i + 1) % 3 == 0]  # 使用self.hidden_sizes
        
        # 初始化池化、激活、展平
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2).to(self.device)
        self.act = nn.ReLU().to(self.device)
        self.flatten = nn.Flatten().to(self.device)

        # 初始化卷积层（自动添加残差路径）
        in_ch = self.input_channels
        for layer_idx, out_ch in enumerate(self.hidden_sizes):  # 使用self.hidden_sizes
            current_layer = nn.ModuleList()
            
            # 主路径：Conv + BN
            current_layer.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1).to(self.device))
            current_layer.append(nn.BatchNorm2d(out_ch).to(self.device))
            
            # 如果当前层需要残差连接，添加1x1卷积
            if layer_idx in self.residual_layers:
                residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1).to(self.device)
                nn.init.kaiming_normal_(residual_conv.weight, mode='fan_out', nonlinearity='relu')
                current_layer.append(residual_conv)
            
            self.layers.append(current_layer)
            in_ch = out_ch
        
        self._update_fc_layer()

    def _update_residual_layers(self):
        """更新残差层索引"""
        self.residual_layers = [i for i in range(len(self.hidden_sizes)) if (i + 1) % 3 == 0]
    
    def compute_fisher(self, dataloader):
        """计算当前任务的Fisher信息矩阵"""
        self.eval()
        fisher_dict = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        
        for inputs, _ in dataloader:
            inputs = inputs.to(self.device)
            self.zero_grad()
            outputs = self(inputs)
            loss = F.cross_entropy(outputs, outputs.argmax(dim=1))  # 无监督代理损失
            loss.backward()
            
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher_dict[n] += p.grad.detach().clone() ** 2
        
        # 归一化并存储
        self.fisher[self.task_count] = {n: f / len(dataloader) for n, f in fisher_dict.items()}
        self.ewc_params[self.task_count] = {n: p.detach().clone() for n, p in self.named_parameters()}

    def ewc_loss(self):
        """计算所有已学任务的EWC正则化损失"""
        loss = 0
        for task in range(self.task_count):
            for n, p in self.named_parameters():
                if n in self.fisher[task] and n in self.ewc_params[task]:
                    loss += (self.fisher[task][n] * 
                            (p - self.ewc_params[task][n]) ** 2).sum()
        return loss

    def update_plasticity_mask(self, decay_factor=0.3):
        """根据参数变化幅度更新梯度掩码"""
        for n, p in self.named_parameters():
            if n not in self.plasticity_mask:
                self.plasticity_mask[n] = torch.ones_like(p)
            
            # 计算参数变化量（相对于初始值）
            delta = torch.abs(p.detach() - self.ewc_params.get(0, {}).get(n, p.detach()))
            self.plasticity_mask[n] = torch.exp(-decay_factor * delta)

    def apply_gradient_mask(self):
        """在反向传播后应用可塑性掩码"""
        for n, p in self.named_parameters():
            if p.grad is not None and n in self.plasticity_mask:
                p.grad *= self.plasticity_mask[n]
                
    def _update_fc_layer(self):
        """动态计算全连接层输入维度（严格对齐forward流程）"""
        with torch.no_grad():
            dummy_input = torch.randn(1, self.input_channels, self.size, self.size).to(self.device)
            for layer_idx, layer_group in enumerate(self.layers):
                identity = dummy_input.clone()
                
                # 主路径：Conv -> BN
                x = layer_group[0](dummy_input)
                x = layer_group[1](x)
                
                # 残差路径（如果启用）
                if layer_idx in self.residual_layers:
                    residual = layer_group[2](identity)
                    x += residual  # 残差相加
                
                # 激活 + 池化
                x = self.act(x)
                x = self.pool(x)
                dummy_input = x
            
            in_features = dummy_input.numel() // dummy_input.shape[0]
            self.fc = nn.Linear(in_features, self.num_class).to(self.device)

    def forward(self, x):
        for layer_idx, layer_group in enumerate(self.layers):
            identity = x.clone()
            
            # 主路径：Conv -> BN
            x = layer_group[0](x)
            x = layer_group[1](x)
            
            # 残差路径（如果启用）
            if layer_idx in self.residual_layers:
                residual = layer_group[2](identity)
                x += residual  # 残差相加
            
            # 激活 + 池化
            x = self.act(x)
            x = self.pool(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def modify_channels(self, layer_index, increase_channel):
        """修改指定层通道数（同步更新残差路径）"""
        assert 0 <= layer_index < len(self.layers), "Invalid layer index"
        
        # 修改主卷积层
        main_conv = self.layers[layer_index][0]
        new_main_conv = nn.Conv2d(
            main_conv.in_channels,
            main_conv.out_channels + increase_channel,
            kernel_size=3, 
            padding=1
        ).to(self.device)
        new_main_conv.weight.data[:main_conv.out_channels] = main_conv.weight.data
        self.layers[layer_index][0] = new_main_conv
        
        # 更新BN层
        new_bn = nn.BatchNorm2d(new_main_conv.out_channels).to(self.device)
        self.layers[layer_index][1] = new_bn
        
        # 如果该层有残差连接，更新残差卷积
        if layer_index in self.residual_layers:
            residual_conv = self.layers[layer_index][2]
            new_residual_conv = nn.Conv2d(
                residual_conv.in_channels,
                new_main_conv.out_channels,  # 输出通道与主路径同步
                kernel_size=1
            ).to(self.device)
            new_residual_conv.weight.data[:residual_conv.out_channels] = residual_conv.weight.data
            self.layers[layer_index][2] = new_residual_conv
        
        # 更新下一层的输入通道
        if layer_index + 1 < len(self.layers):
            next_conv = self.layers[layer_index + 1][0]
            new_next_conv = nn.Conv2d(
                new_main_conv.out_channels,
                next_conv.out_channels,
                kernel_size=3,
                padding=1
            ).to(self.device)
            new_next_conv.weight.data[:, :next_conv.in_channels] = next_conv.weight.data
            self.layers[layer_index + 1][0] = new_next_conv
        
        self._update_fc_layer()  # 修改后更新全连接层
        self._update_residual_layers()  # 修改后更新索引

    # -------------------------- 关键修改2：修正合并子层逻辑 --------------------------
    def merge_sub_layers(self, layer_index):
        if len(self.layers[layer_index]) == 2:
            return
        
        # 获取两个子卷积层和对应的BN层
        conv1 = self.layers[layer_index][0]
        conv2 = self.layers[layer_index][1]
        bn = self.layers[layer_index][2]
        
        # 合并后的输入通道是conv1.in_channels，输出是conv1.out_channels + conv2.out_channels
        merge_conv = nn.Conv2d(conv1.in_channels, conv1.out_channels + conv2.out_channels, 
                              kernel_size=3, padding=1).to(self.device)
        merge_conv.weight.data[:conv1.out_channels] = conv1.weight.data
        merge_conv.weight.data[conv1.out_channels:] = conv2.weight.data
        
        # 合并BN层参数（假设原BN层已经处理了合并后的通道）
        self.layers[layer_index] = nn.ModuleList([merge_conv, bn])
        self._update_fc_layer()  # 合并后更新全连接层
        self._update_residual_layers()  # 修改后更新索引

    def add_conv_layer(self, new_out_channels):
        prev_layer = self.layers[-1][0]
        new_conv = nn.Conv2d(prev_layer.out_channels, new_out_channels, 
                            kernel_size=3, padding=1).to(self.device)
        new_bn = nn.BatchNorm2d(new_out_channels).to(self.device)
        self.layers.append(nn.ModuleList([new_conv, new_bn]))
        self._update_fc_layer()  # 添加层后更新全连接层
        self._update_residual_layers()  # 修改后更新索引


def train(model, optimizer, epochs=2, ewc_lambda=1e4):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 假设train_loader是当前任务的数据加载器
        for x, y in train_loader:  # Ensure train_loader is defined before this loop
            x, y = x.to(model.device), y.to(model.device)
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)
            
            # 🚀 添加EWC正则项（排除当前任务）
            if model.task_count > 0:
                loss += ewc_lambda * model.ewc_loss()
            
            # 反向传播与梯度掩码
            loss.backward()
            model.apply_gradient_mask()  # 🚀 应用掩码
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # 🚀 任务结束后更新Fisher信息
    model.compute_fisher(train_loader)
    model.task_count += 1

if __name__ == '__main__':
    input_channels = 3  # 输入为3通道图像
    in_size = 32
    num_class = 10
    conv_layers = [16]  # 现有卷积层
    bs = 16

    net = DynamicCNN(input_channels=input_channels, conv_hidden_layers=conv_layers, num_classes=num_class).cuda()

    # Define train_loader with a dataset (e.g., CIFAR-10)
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    # 使用示例
    x = torch.randn(bs, input_channels, in_size, in_size).cuda()  # 假设输入为5个28x28的图片
    y = torch.randint(0, num_class, (bs,)).cuda()

    # 训练设置
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    print(net)
    train(optimizer=optimizer)

    net.modify_channels(layer_index=len(net.layers) - 1, increase_channel=24)
    print(net)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    train(optimizer=optimizer)
    net.merge_sub_layers(layer_index=len(net.layers) - 1)
    print(net)

    net.modify_channels(layer_index=len(net.layers) - 1, increase_channel=24)
    print(net)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    train(optimizer=optimizer)
    net.merge_sub_layers(layer_index=len(net.layers) - 1)
    print(net)

    # =========================== 添加卷积层，假设输出通道为24 ====================
    net.add_conv_layer(new_out_channels=25)
    print(net)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    train(optimizer=optimizer)
    net.merge_sub_layers(layer_index=len(net.layers) - 1)
    print(net)
    
    net.modify_channels(layer_index=len(net.layers) - 1, increase_channel=24)
    print(net)
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    train(optimizer=optimizer)
    net.merge_sub_layers(layer_index=len(net.layers) - 1)
    print(net)

    # net.add_conv_layer(layer_index=2, new_out_channels=50)
    # print(net)
    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    # train(optimizer=optimizer, epochs=5)
