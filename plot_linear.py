import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def draw_network_digraph(input_num, task=1):
    G = nx.DiGraph()
    
    node_num = 0
    # 隐层---隐层
    for layer in range(len(input_num) - 1):
        for i in range(input_num[layer]):
            for j in range(input_num[layer + 1]):
                G.add_edge(node_num + i, node_num + input_num[layer] + j, weight=np.random.uniform(1, 10))
        node_num += input_num[layer]
    
    # =========================================================================================
    pos = dict()
    
    num = 0
    index = 0
    for layer in range(len(input_num)):
        for i in range(0, input_num[layer]):
            tmp = i + num
            pos[tmp] = (index, i - input_num[layer] / 2)
            
        num += input_num[layer]
        index += 1
    
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    pre_colors = ['red', 'green', 'blue', 'cyan', 'purple']
    node_colors = []
    num_nodes = G.number_of_nodes()
    
    for i in range(num_nodes):
        node_colors.append('black')
        
    # index = np.arange(26, 47)
    # for i in index:
    #     node_colors[i] = 'red'
    
    # index = np.arange(26, 48)
    # for i in index:
    #     node_colors[i] = 'red'
        
    plt.figure(figsize=(12 ,8))
    nx.draw(G,
        pos,
        # with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        width=0.5,
        node_size=60,
        edge_cmap=plt.cm.inferno, # viridis  inferno  magma  cividis
        alpha=0.5)
    # plt.colorbar()
    # plt.show()
    plt.savefig(f'task_{task}.png', dpi=300)

        
if __name__ == '__main__':
    draw_network_digraph([1, 10, 10], task=0)
    # draw_network_digraph([1, 20, 10], task=1)
    # draw_network_digraph([1, 25, 22, 10], task=2)
    # draw_network_digraph([1, 25, 26, 10], task=3)
    # draw_network_digraph([1, 25, 46, 10], task=4)
    # draw_network_digraph([1, 25, 66, 44, 10], task=5)
    