import warnings
warnings.filterwarnings('ignore')

from torchvision import datasets
from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pandas as pd
import logging
from torch.utils.data import random_split
import yaml
from datetime import datetime
from cutout import Cutout
from linear_add import DynamicLinearNet
from utils import train, test, set_dead_neuron, compute_average_weight_magnitude
from load_cifar_data import CustomCIFARDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="self replicating network for linear and CNN")
    # ---------------- hyper params ------------------
    parser.add_argument('--description', type=str, default='train the model from last training when add layers, set the params learnable')
    parser.add_argument('--lr', type=float, default='0.0001', help='')
    parser.add_argument('--momentum', type=float, default='0.95', help='')
    parser.add_argument('--weight_decay', type=float, default='0.001', help='')
    parser.add_argument('--bs', type=int, default='512', help='')
    parser.add_argument('--epoches', type=int, default='5', help='')
    parser.add_argument('--max_growth_times', type=int, default='2', help='max increase times add layers')
    parser.add_argument('--improve_acc', type=float, default='0.003', help='')
    parser.add_argument('--resize', type=int, default='32', help='')
    parser.add_argument('--init_neurons', nargs='+', type=int, default=[100], help='init hidden neurons')
    parser.add_argument('--increase_neurons', type=int, default='50', help='increase neurons')
    parser.add_argument('--dead_prop', type=float, default='0.5', help='')
    parser.add_argument('--dead_mode', default='None',  
                        choices=['on_neuron_zero_weight_no_learnable', 'on_connect_fix_weight_no_learnable', 'on_neuron_zero_weight_learnable', 
                                 'on_connect_random_weight_learnable', 'None', 'on_connect_zero_weight_learnable'])

    parser.add_argument('--dataset', type=str, default='mnist', choices=['cifar10', 'cifar100', 'mnist', 'fashionmnist'], help='the max train times for a new model')
    parser.add_argument('--model', type=str, default='linear', help='the model type')
    args = parser.parse_args()
    
    # ------------------------------------------------------------------------------------------------------------
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d===%H-%M-%S')
    save_root = f'./results/{args.dataset}/{args.model}/{formatted_time}'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    args_dict = vars(args)
    with open(os.path.join(save_root, 'config.yaml') , 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)
        
    # --------------- log记录 --------------------------------
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(save_root, f'{args.model}_{args.dataset}.log'), mode='w'),  # 日志文件
                        logging.StreamHandler()           # 控制台输出
                    ])

    logging.warning(args)
    
    # -------------------------------- 数据集加载 ----------------------------------------------------------------------------
    # transform_list = [
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ]
    # transform_list.append(transforms.Lambda(lambda x: x.view(-1))) 
    # transform = transforms.Compose(transform_list)
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(args.resize, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    Cutout(n_holes=1, length=16)
])
            
    if args.dataset == 'cifar10':
        trainset = datasets.CIFAR10(root='/data4/zpf/dataset/cifar10', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='/data4/zpf/dataset/cifar10', train=False, download=True, transform=transform)
        num_classes = len(trainset.classes)
        num_class_per_task = 2
        input_size = 32 * 32 * 3

        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        trainset, valset = random_split(trainset, [train_size, val_size])
    elif args.dataset == 'cifar100':
        trainset = datasets.CIFAR100(root='/data4/zpf/dataset/cifar100', train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root='/data4/zpf/dataset/cifar100', train=False, download=True, transform=transform)
        num_classes = len(trainset.classes)
        num_class_per_task = 5
        input_size = 32 * 32 * 3

        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        trainset, valset = random_split(trainset, [train_size, val_size])
    elif args.dataset == 'mnist':
        transform_mnist = transforms.Compose([
            transforms.Lambda(lambda x: x.unsqueeze(0)),
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize((0.3081,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
            ])

        trainset = datasets.MNIST(root='D:/Dataset/public/MNIST', train=True, download=True, transform=None)
        testset = datasets.MNIST(root='D:/Dataset/public/MNIST', train=False, download=True, transform=None)
        num_classes = len(trainset.classes)
        num_class_per_task = 2
        input_size = 28 * 28

        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        trainset, valset = random_split(trainset, [train_size, val_size])
    elif args.dataset == 'fashionmnist':
        transform_mnist = transforms.Compose([
            transforms.Lambda(lambda x: x.unsqueeze(0)),
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize((0.3081,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
            ])

        trainset = datasets.FashionMNIST(root='D:/Dataset/public/Fashion-MNIST', train=True, download=True, transform=None)
        testset = datasets.FashionMNIST(root='D:/Dataset/public/Fashion-MNIST', train=False, download=True, transform=None)
        num_classes = len(trainset.classes)
        num_class_per_task = 2
        input_size = 28 * 28

        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        trainset, valset = random_split(trainset, [train_size, val_size])
    
    # -------------------------- 固定随机种子 -----------------------------
    seed = 20151103
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    class_idx = np.arange(num_classes)
    shuffled_class_idx = np.random.permutation(class_idx)
    
    # ---------------------------- 模型、优化器、损失函数 --------------------------------------------------------------------------------
    model = DynamicLinearNet(input_size=input_size, output_size=num_classes, hidden_sizes=args.init_neurons).cuda()
    save_fig_name = f'linear_{args.dataset}.png'
    save_csv_name = f'linear_{args.dataset}.csv'
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    # ------------------------------ train and test -----------------------------------------
    target_class = []
    acc = []
    valid_train_acces = []
    valid_train_losses = []
    valid_val_acces = []
    epoch_split = []
    train_times_split = []
    average_weight_magnitude = []
    mask = None
    
    for task in range(num_classes // num_class_per_task):
        task_train_acces = []
        task_train_losses = []
        task_val_acces = []
        task_average_weight_magnitude = []
        
        task_train_times = 1
        
        task_val_acc = 0
        task_best_val_acc = 0
        task_best_epoch = 0
        task_best_train_times = 0
        task_layer_add_times = 0
        task_epoch_total = 0
        task_add_layers_train_times = []

        target_class.extend(shuffled_class_idx[task * num_class_per_task + j] for j in range(num_class_per_task))
        RED = '\033[31m'
        RESET = '\033[0m'  # 重置为默认颜色                
        print(f'{RED}target_class: {target_class}{RESET}')
        
        train_dataset = CustomCIFARDataset(trainset, target_class, transform=transform)
        val_dataset = CustomCIFARDataset(valset, target_class, transform=transform)
        test_dataset = CustomCIFARDataset(testset, target_class, transform=transform)
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True)
        
        while True:
            logging.info(f'---------------------- [Task: {task + 1}, train_times: {task_train_times}] --------------------')
            train_acc, train_loss, val_acc = train(net=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                                                    optimizer=optimizer, criterion=criterion, 
                                                    epochs=args.epoches,
                                                    mask=mask, model_type='linear')
            mask = None
            task_epoch_total += args.epoches 
            task_train_times += 1
            
            task_val_acces.append(val_acc.cpu().item())
            task_train_acces.extend(tmp.cpu().item() for tmp in train_acc)
            task_train_losses.extend(tmp for tmp in train_loss)
            task_average_weight_magnitude.append(compute_average_weight_magnitude(model))

            model.merge_sub_layers(layer_index=len(model.layers) - 2)
            logging.info(model)
            
            # 生长超出最大次数仍然没有长进，退出生长
            if task_layer_add_times >= args.max_growth_times:
                logging.info('----- Replicating Over -----')
                break
            
            # 记录在val data上的最好性能
            if val_acc - task_best_val_acc > args.improve_acc:
                task_best_val_acc = val_acc
                task_layer_add_times = 0
                torch.save(model, os.path.join(save_root, f'best_model_{task}.pth'))
                task_best_epoch = task_epoch_total
                task_best_train_times = task_train_times
                
                # 网络生长 
                logging.info(f'----- Adding linear neurons -----')
                model.modify_neurons(layer_index=len(model.layers) - 2, num_neurons=args.increase_neurons)
            else:  # 深度生长
                if len(model.layers) <= 3:
                    model.add_layer(num_neurons=int(model.layers[len(model.layers) - 2][0].in_features * 1 / 4))
                logging.info(f'----- Adding linear layers {task_layer_add_times + 1} -----')
                task_layer_add_times += 1

            # 网络每修改一次，在训练前重新设置dead neuron   
            # logging.info(f'^^^^^ set dead neuron {args.dead_prop} {args.dead_mode} ^^^^^')
            # mask = set_dead_neuron(model=model, dead_prop=args.dead_prop, mode=args.dead_mode, model_type='linear')
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        # 只记录有效的训练数据，即best_val_acc之前的数据
        valid_train_acces.extend(task_train_acces[:task_best_epoch])
        valid_train_losses.extend(task_train_losses[:task_best_epoch])
        valid_val_acces.extend(task_val_acces[:(task_best_train_times - 1)])
        average_weight_magnitude.extend(task_average_weight_magnitude[:(task_best_train_times - 1)])
        epoch_split.append((task_best_epoch + epoch_split[-1]) if len(epoch_split) else task_best_epoch)
        train_times_split.append((task_best_train_times - 1 + train_times_split[-1]) if len(train_times_split) else (task_best_train_times - 1))
        
        # ========================= 生长结束 ==================================
        # 在测试集上测试结果
        model = torch.load(os.path.join(save_root, f'best_model_{task}.pth'))
        logging.info(model)
        test_acc = test(model, test_dataloader)
        acc.append(test_acc.cpu().item())
        
        logging.info(f'\n\n********************* Task: {task + 1}, Test Acc: {test_acc:.3f} *********************\n\n')
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        # 每个task结束后，会重新修改模型，加载当前task在val最好的性能对应的权重
        logging.info(f'^^^^^ set dead neuron {args.dead_prop} {args.dead_mode} ^^^^^')
        mask = set_dead_neuron(model=model, dead_prop=args.dead_prop, mode=args.dead_mode, model_type='linear')
        
    logging.info(acc)
    # -------------------------------------- save fig ----------------------------------------------------------------------
    plt.figure(figsize=(12, 9))
    x_index = np.arange(1, len(valid_train_acces) + 1)
    
    plt.subplot(2, 2, 1)
    plt.plot(x_index, valid_train_acces, marker='*', markersize=3)
    for v in epoch_split:
        plt.axvline(x=v, color='r', linestyle='--', alpha=0.2)
    plt.title('train accuracy')
    
    plt.subplot(2, 2, 3)
    plt.plot(x_index, valid_train_losses, marker='*', markersize=3)
    for v in epoch_split:
        plt.axvline(x=v, color='r', linestyle='--', alpha=0.2)
    plt.title('train loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(valid_val_acces, marker='*', markersize=3)
    for v in train_times_split:
        plt.axvline(x=v - 1, color='r', linestyle='--', alpha=0.2)
    plt.title('val accuracy')
    
    plt.subplot(2, 2, 4)
    plt.plot(acc, marker='*', markersize=3)
    plt.title('test accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, save_fig_name), dpi=300)
    
    # ==============
    plt.figure()
    plt.plot(average_weight_magnitude)
    for v in train_times_split:
        plt.axvline(x=v - 1, color='r', linestyle='--', alpha=0.2)
    plt.title('average_weight_magnitude')
    plt.xlabel('tasks increase')
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, 'average_weight_magnitude.png'), dpi=300)
    
    # ----------------------------------------- 保存结果数据 ---------------------------------
    PlotData = {
        'valid_train_acces': valid_train_acces,
        'valid_train_losses': valid_train_losses,
        'valid_val_acces': valid_val_acces,
        'test_acces': acc,
        'average_weight_magnitude': average_weight_magnitude
    }

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in PlotData.items()]))
    df.to_csv(os.path.join(save_root, save_csv_name), index=False)
    