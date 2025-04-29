import warnings

warnings.filterwarnings('ignore')

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.utils.prune as prune 
from utils import test, train_demo, get_data, ToyConv, ToyLinear
from load_cifar_data import CustomCIFARDataset
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from models import AlexNet


if __name__ == '__main__':
    # -------------------------------- 数据集加载 ----------------------------------------------------------------------------
    batch_size = 512
    lr = 0.005
    momentum = 0.95
    weight_decay = 0.05
    epoch = 30
    resize = 128
    
    # train_dataloader, test_dataloader, num_classes = get_data(data_name='cifar100', flatten=True, bs=batch_size)
    
    transform_list = [
        transforms.ToTensor(),
        transforms.Resize(resize),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    # transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(transform_list)
    # ------------------------------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # ------------------------------------------------
    
    
    trainset = datasets.CIFAR100(root='D:/Dataset/public/cifar100', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root='D:/Dataset/public/cifar100', train=False, download=True, transform=transform_test)
    num_classes = len(trainset.classes)
    num_class_per_task = 100

    hidden = [1100]
    # model = ToyLinear(input_size=32 * 32 * 3, output_size=num_classes, hidden_sizes=hidden).cuda()
    model = ToyConv(3, [23, 95, 73], num_classes, size=32).cuda()
    # model = AlexNet(num_classes=100).cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # criterion = torch.nn.CrossEntropyLoss()
    # print(model)
    # print(f'batch_size = {batch_size} lr = {lr} momentum = {momentum} weight_decay = {weight_decay} epoch = {epoch}')
    
    # # -------------------------- 固定随机种子 -----------------------------
    # np.random.seed(2024)
    # torch.manual_seed(2024)
    # torch.cuda.manual_seed(2024)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # class_idx = np.arange(num_classes)
    # shuffled_class_idx = np.random.permutation(class_idx)
    
    # target_class = []
    # test_task_acc = []
    
    # for task in range(num_classes // num_class_per_task):
    #     # target_class = []
    #     target_class.extend(shuffled_class_idx[task * num_class_per_task + j] for j in range(num_class_per_task))
    #     RED = '\033[31m'
    #     RESET = '\033[0m'  # 重置为默认颜色                
    #     print(f'{RED}target_class: {target_class}{RESET}')
    #     print('*' * 50, f'Task {task + 1}', '*' * 50)
        
    #     train_dataset = CustomCIFARDataset(trainset, target_class, transform=transform)
    #     test_dataset = CustomCIFARDataset(testset, target_class, transform=transform)
        
    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    #     # ------------------------------ train and test -----------------------------------------
    #     train_acc, train_loss, test_acc = train_demo(net=model, train_dataloader=train_dataloader,
    #                                             test_dataloader=test_dataloader,
    #                                             optimizer=optimizer, criterion=criterion, 
    #                                             epochs=epoch,
    #                                             target_class=target_class)
        