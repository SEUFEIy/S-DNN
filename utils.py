import torch
import logging
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np


# --------------------------------------------------------------------------------------------------------------------------
def train(net, train_dataloader, val_dataloader, optimizer, criterion, epochs, mask=None, model_type='linear'):
    train_acces = []
    train_losses = []
    
    for epoch in range(epochs):
        train_correct = 0
        train_total = 0
        train_loss = 0
        
        # ---------------- train ----------------------
        for images, labels in train_dataloader:
            images, labels = images.cuda(), labels.cuda()
            
            pred = net(images)
            loss = criterion(pred, labels)        
            
            optimizer.zero_grad()
            loss.backward()
            
            # ++++++++++++++++++++++++++++++
            if mask is not None:
                apply_grad(model=net, mask=mask, model_type=model_type)
            # ++++++++++++++++++++++++++++++
            
            train_loss += loss.item()
            optimizer.step()
            
            train_total += images.size(0)
            train_correct += torch.sum(pred.argmax(axis=1) == labels)
        
        train_acces.append(train_correct / train_total)
        train_losses.append(train_loss / train_total)
    
        logging.info(f'Epoch:{epoch + 1} TrainLoss:{train_loss / train_total:.3f} TrainAcc:{train_correct / train_total:.3f}')
       
    # -------------------- test ---------------------
    val_correct = 0
    val_total = 0
    
    for images, labels in val_dataloader:
        images, labels = images.cuda(), labels.cuda()
        
        with torch.no_grad():
            pred = net(images)
            
            val_total += images.size(0)
            val_correct += torch.sum(pred.argmax(axis=1) == labels)
    
    val_acc = val_correct / val_total
    
    logging.info(f'==================== ValAcc: {val_acc:.3f} ==================')
    
    return train_acces, train_losses, val_acc


# --------------------------------------------------------------------------------------------------------------------------
def train_demo(net, train_dataloader, test_dataloader, optimizer, criterion, epochs, target_class):
    train_acces = []
    train_losses = []
    
    for epoch in range(epochs):
        train_correct = 0
        train_total = 0
        train_loss = 0
        
        # ---------------- train ----------------------
        for images, labels in train_dataloader:
            images, labels = images.cuda(), labels.cuda()
            
            pred = net(images)
            loss = criterion(pred, labels)        
            
            optimizer.zero_grad()
            loss.backward()
            
            train_loss += loss.item()
            optimizer.step()
            
            train_total += images.size(0)
            train_correct += torch.sum(pred.argmax(axis=1) == labels)
        
        train_acces.append(train_correct / train_total)
        train_losses.append(train_loss / train_total)
    
        print(f'Epoch:{epoch + 1} TrainLoss:{train_loss / train_total:.3f} TrainAcc:{train_correct / train_total:.3f}', end=' ')
       
        # -------------------- test ---------------------
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []
        
        for images, labels in test_dataloader:
            images, labels = images.cuda(), labels.cuda()
            
            with torch.no_grad():
                pred = net(images)
                
                _, preds = torch.max(pred, 1)  # 获取预测标签
                all_labels.extend(labels.cpu().numpy())  # 将真实标签添加到列表
                all_preds.extend(preds.cpu().numpy())  # 将预测标签添加到列表
                
                val_total += images.size(0)
                val_correct += torch.sum(pred.argmax(axis=1) == labels)
        
        val_acc = val_correct / val_total
    
    # # 将列表转换为 NumPy 数组
    # all_labels = np.array(all_labels)
    # all_preds = np.array(all_preds)
    
    # # 计算每个类别的精度
    # accuracies = {}

    # for class_index in target_class:
    #     # 取出每个类别的索引
    #     class_mask = (all_labels == class_index)
    #     class_accuracy = accuracy_score(all_labels[class_mask], all_preds[class_mask]) if class_mask.any() else 0
    #     accuracies[class_index] = class_accuracy
    
    # # 打印每个类别的精度
    # for class_index, accuracy in accuracies.items():
    #     print(f'Class {class_index}, Accuracy: {accuracy:.2f}')
    
    # print(f'==================== TestAcc: {val_acc:.3f} ==================')
        print(f'TestAcc: {val_acc}')
    
    return train_acces, train_losses, val_acc


# --------------------------------------------------------------------------------------------------------------------------
def test(net, test_dataloader,):
    test_correct = 0
    test_total = 0

    for images, labels in test_dataloader:
        images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            pred = net(images)

            test_total += images.size(0)
            test_correct += torch.sum(pred.argmax(axis=1) == labels)

    test_acc = test_correct / test_total
    
    logging.info(f'==================== TestAcc: {test_acc:.3f} ==================')
    
    return test_acc


# --------------------------------------------------------------------------------------------------------------------------
def get_data(data_name, flatten, bs):
    transform_list = [
        transforms.ToTensor(),
        # transforms.Resize(128),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    if data_name == 'cifar10':
        trainset = datasets.CIFAR10(root='/data4/zpf/dataset/cifar10', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='/data4/zpf/dataset/cifar10', train=False, download=True, transform=transform)
    elif data_name == 'cifar100':
        trainset = datasets.CIFAR100(root='/data4/zpf/dataset/cifar100', train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root='/data4/zpf/dataset/cifar100', train=False, download=True, transform=transform)
    elif 'mnist'in data_name:
        transform_mnist = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.unsqueeze(0)),
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.3081,), (0.3081,)),
            # transforms.Lambda(lambda x: x.view(-1))
            ])
        
        trainset = datasets.MNIST(root='D:/Dataset/public/MNIST', train=True, download=True, transform=transform_mnist)
        testset = datasets.MNIST(root='D:/Dataset/public/MNIST', train=False, download=True, transform=transform_mnist)
        
    num_classes = len(trainset.classes)

    train_dataloader = DataLoader(trainset, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=bs, shuffle=False)
    
    return train_dataloader, test_dataloader, num_classes


# --------------------------------------------------------------------------------------------------------------------------
class ToyConv(nn.Module):
    def __init__(self, in_ch, hid_ch, num_classes, device='cuda', size=32):
        super(ToyConv, self).__init__()
        
        self.layers = nn.ModuleList()
        self.BN_layers = nn.ModuleList()
        
        self.input_size = in_ch
        self.output_size = num_classes
        self.hidden_size = hid_ch
        self.device = torch.device(device=device)
         
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2).to(self.device)
        
        for out_ch in self.hidden_size:
            self.layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1).to(device))
            self.BN_layers.append(nn.BatchNorm2d(out_ch))
            in_ch = out_ch
        
        # 添加分类头
        w_h = int(size / (2 ** len(self.layers)))
        self.fc = nn.Linear(out_ch * w_h * w_h , self.output_size).to(self.device)
        # self.fc = nn.Sequential(
        #     nn.Linear(out_ch * w_h * w_h , 1000).to(self.device),
        #     nn.Linear(1000 , 500).to(self.device),
        #     nn.Linear(500 , self.output_size).to(self.device))
    
    def forward(self, x):
        for layer, BN_layer in zip(self.layers, self.BN_layers):
            x = self.pool(F.relu(BN_layer(layer(x))))
        
        x = self.fc(x.view(x.size(0), -1))

        return x
        

# --------------------------------------------------------------------------------------------------------------------------
class ToyLinear(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, device='cuda'):
        super(ToyLinear, self).__init__()

        self.layers = nn.ModuleList()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.device = torch.device(device=device)

        if hidden_sizes:
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]).to(device))

            for i in range(1, len(hidden_sizes)):
                self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]).to(device))

            self.layers.append(nn.Linear(hidden_sizes[-1], output_size).to(device))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        x = self.layers[-1](x)

        return x
    
    
# --------------------------------------------------------------------------------------------------------------------------
def set_dead_neuron(model, dead_prop, mode, model_type='linear'):
    if mode == 'None':
        return None
    
    if model_type == 'linear':
        mask = []
        
        for i in range(len(model.layers)):
            for j in range(len(model.layers[i])):
                with torch.no_grad():
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    if mode == 'on_neuron_zero_weight_no_learnable': 
                        """ 
                            以神经元为单位, 指定神经元失效, 所有权重为0, 相当于直接砍掉这些神经元
                            网络规模减小 计算量减小
                        """
                        mask_layer = torch.rand(model.layers[i][j].out_features) > dead_prop              
                        false_index = torch.nonzero(~mask_layer)
                        model.layers[i][j].weight.data[false_index, :] = torch.zeros(model.layers[i][j].in_features).cuda()
                        mask_layer = mask_layer.unsqueeze(1).expand(-1, model.layers[i][j].in_features)
                        mask.append(mask_layer)
                        
                        dead_num = mask_layer.numel() - torch.sum(mask_layer).item()
                        total_num = mask_layer.numel()
                        print(f'mask ratio:  {dead_num} / {total_num} = {dead_num / total_num:.3f}')
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    elif mode == 'on_connect_fix_weight_no_learnable':
                        """ 
                            以神经元的连接权重为单位, 指定连接权重固定住, 从此不再更新
                            网络规模不变 可学习的神经元数量减少
                        """
                        mask_layer = torch.rand(model.layers[i][j].out_features, model.layers[i][j].in_features) > dead_prop
                        mask.append(mask_layer)
                        
                        dead_num = mask_layer.numel() - torch.sum(mask_layer).item()
                        total_num = mask_layer.numel()
                        print(f'mask ratio:  {dead_num} / {total_num} = {dead_num / total_num:.3f}')
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    elif mode == 'on_neuron_zero_weight_learnable':
                        """ 
                            将神经元的所有链接置0, 但是可以接着更新
                        """
                        mask_layer = torch.rand(model.layers[i][j].out_features) > dead_prop              
                        false_index = torch.nonzero(~mask_layer)
                        model.layers[i][j].weight.data[false_index, :] = torch.zeros(model.layers[i][j].in_features).cuda()
                        mask = None
                        
                        dead_num = mask_layer.numel() - torch.sum(mask_layer).item()
                        total_num = mask_layer.numel()
                        print(f'mask ratio:  {dead_num} / {total_num} = {dead_num / total_num:.3f}')
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    elif mode == 'on_connect_random_weight_learnable':
                        """ 
                            将神经元部分链接置0, 但是可以接着更新
                        """
                        mask_layer = torch.rand(model.layers[i][j].out_features, model.layers[i][j].in_features) > dead_prop
                        mask = None
                        false_index = torch.nonzero(~mask_layer)
                    
                        for k in false_index:
                            model.layers[i][j].weight.data[k[0].item()][k[1].item()] = torch.randn(1).cuda()
                        
                        dead_num = mask_layer.numel() - torch.sum(mask_layer).item()
                        total_num = mask_layer.numel()
                        print(f'mask ratio:  {dead_num} / {total_num} = {dead_num / total_num:.3f}')
                    elif mode == 'on_connect_zero_weight_learnable':
                        """ 
                            将神经元部分链接置0, 但是可以接着更新
                        """
                        mask_layer = torch.rand(model.layers[i][j].out_features, model.layers[i][j].in_features) > dead_prop
                        mask = None
                        false_index = torch.nonzero(~mask_layer)
                    
                        for k in false_index:
                            model.layers[i][j].weight.data[k[0].item()][k[1].item()] = torch.zeros(1).cuda()
                        
                        dead_num = mask_layer.numel() - torch.sum(mask_layer).item()
                        total_num = mask_layer.numel()
                        print(f'mask ratio:  {dead_num} / {total_num} = {dead_num / total_num:.3f}')
                    else:
                        raise ValueError('invalid mode of linear in function set_dead_neuron')
                    
        return mask
    # *****************************************************************************************
    elif model_type == 'cnn':
        mask = []
        
        for i in range(len(model.layers)):
            for j in range(len(model.layers[i])):
                if isinstance(model.layers[i][j], nn.Conv2d):
                    
                    with torch.no_grad():
                        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        if mode == 'on_channel_zero_weight_no_learnable': 
                            """ 
                                以卷积核的通道为单位, 指定输出通道失效, 所有权重为0, 相当于直接砍掉这一个输出通道
                                网络规模减小 计算量减小
                            """
                            t_shape = model.layers[i][j].weight.shape
                            mask_layer = torch.rand(model.layers[i][j].out_channels) > dead_prop              
                            false_index = torch.nonzero(~mask_layer)
                            model.layers[i][j].weight.data[false_index, :] = torch.zeros(t_shape[1], t_shape[2], t_shape[3]).cuda()
                            
                            t = torch.zeros((t_shape[0], t_shape[1], t_shape[2], t_shape[3]), dtype=torch.bool)
                            for j in range(t.shape[0]):
                                t[j] = mask_layer[j]
                            mask.append(t)
                            
                            dead_num = t.numel() - torch.sum(t).item()
                            total_num = t.numel()
                            print(f'mask ratio:  {dead_num} / {total_num} = {dead_num / total_num:.3f}')
                        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        elif mode == 'on_connect_fix_weight_no_learnable':
                            """ 
                                以神经元的连接权重为单位, 指定连接权重固定住, 从此不再更新
                                网络规模不变 可学习的神经元数量减少
                            """
                            mask_layer = torch.rand(model.layers[i][j].weight.shape) > dead_prop
                            dead_num = mask_layer.numel() - torch.sum(mask_layer).item()
                            total_num = mask_layer.numel()
                            print(f'mask ratio:  {dead_num} / {total_num} = {dead_num / total_num:.3f}')
                            mask.append(mask_layer)
                        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        elif mode == 'on_connect_random_weight_learnable':
                            """ 
                                以神经元的连接权重为单位, 指定连接权重固定住, 从此不再更新
                                网络规模不变 可学习的神经元数量减少
                            """
                            mask_layer = torch.rand(model.layers[i][j].weight.shape) > dead_prop
                            model.layers[i][j].weight.data[mask] = torch.tensor(0).cuda()
                            
                            dead_num = mask_layer.numel() - torch.sum(mask_layer).item()
                            total_num = mask_layer.numel()
                            print(f'mask ratio:  {dead_num} / {total_num} = {dead_num / total_num:.3f}')
                            
                            mask = None
                        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        elif mode == 'on_connect_zero_weight_learnable':
                            """ 
                                将神经元的所有链接置0, 但是可以接着更新
                            """
                            mask_layer = torch.rand(model.layers[i][j].weight.shape) > dead_prop
                            mask_layer = mask_layer.float().cuda()
                            model.layers[i][j].weight.data *= mask_layer
                            
                            dead_num = mask_layer.numel() - torch.sum(mask_layer).item()
                            total_num = mask_layer.numel()
                            print(f'mask ratio:  {dead_num} / {total_num} = {dead_num / total_num:.3f}')
                            mask = None
                        else:
                            raise ValueError('invalid mode of cnn in function set_dead_neuron')
        return mask
    # *****************************************************************************************
    else:
        raise ValueError('invalid model_type')
                

# --------------------------------------------------------------------------------------------------------------------------
def apply_grad(model, mask, model_type='linear'):
    index = 0
        
    for i in range(len(model.layers)):
        for j in range(len(model.layers[i])):
            if isinstance(model.layers[i][j], (nn.Linear, nn.Conv2d)):
                model.layers[i][j].weight.grad.mul_(mask[index].cuda())
                index += 1


# --------------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def compute_average_weight_magnitude(net):
    """ Computes average magnitude of the weights in the network """

    num_weights = 0
    sum_weight_magnitude = torch.tensor(0.0, device='cuda')

    for p in net.parameters():
        num_weights += p.numel()
        sum_weight_magnitude += torch.sum(torch.abs(p))

    return sum_weight_magnitude.cpu().item() / num_weights

def get_color():
    color = ['#FFBF00', '#008AEC', 
             '#FD9EB2', '#1975BA',
             '#DA5953', '#45AB84', '#C2B2CD', 
             '#6BCCEF', '#DB74A8', '#00D187', '#FDC848',
             '#482E83', '#F7E488', '#9D3533', '#6BBF9F', '#CE3832', '#587A7E', '#A65350', '#B4338B', '#1D7A9B', '#43793A', '#8E488F', '#C85F28', '#F29D51', '#E37C7B', '#2D3080']

    return color