# # linear cifar10 
#CUDA_VISIBLE_DEVICES=1 python main_linear.py --dataset 'cifar10' --lr 0.001 --epoches 8 --init_neurons 100 --increase_neuron 50 --max_growth_times 5 --dead_prop 0.4 --dead_mode 'on_connect_zero_weight_learnable'

# # linear cifar100
#CUDA_VISIBLE_DEVICES=1 python main_linear.py --dataset 'cifar100' --lr 0.001 --epoches 10 --init_neurons 100 --increase_neuron 50 --max_growth_times 5 --dead_prop 0.4 --dead_mode 'on_connect_random_weight_learnable'


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#CUDA_VISIBLE_DEVICES=1 python main_cnn.py --dataset 'cifar10' --lr 0.001 --epoches 8 --init_channels 8 --increase_channels 8 --max_growth_times 5 --dead_prop 0.4 --dead_mode 'on_connect_zero_weight_learnable'

#CUDA_VISIBLE_DEVICES=2 python main_cnn.py --dataset 'cifar100' --lr 0.001 --epoches 10 --init_channels 8 --increase_channels 5 --max_growth_times 5 --dead_prop 0.4 --dead_mode 'on_connect_zero_weight_learnable' --resize 32

CUDA_VISIBLE_DEVICES=1 python main_cnn.py --dataset 'cifar100' --lr 0.001 --epoches 8 --init_channels 8 --increase_channels 8 --max_growth_times 5 --dead_prop 0.4 --dead_mode 'on_connect_zero_weight_learnable' --seed 1548
#CUDA_VISIBLE_DEVICES=2 python main_cnn.py --dataset 'cifar100' --lr 0.001 --epoches 8 --init_channels 8 --increase_channels 8 --max_growth_times 5 --dead_prop 0.4 --dead_mode 'on_channel_zero_weight_no_learnable'