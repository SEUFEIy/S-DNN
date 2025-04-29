import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def demo01():
    G = nx.DiGraph()
    G.add_edge(1, 3)
    G.add_edge(1, 4)
    G.add_edge(1, 5)

    G.add_edge(2, 4)
    G.add_edge(2, 5)

    pos = {1: (0, 0.25),
        2: (0, 0.75),
        3: (1, 0.2),
        4: (1, 0.5),
        5: (1, 0.8)}
    
    nx.draw(G,
            pos,
            with_labels=True,
            node_color='white',
            edgecolors='black',
            linewidths=3,
            width=2,
            node_size=1000)
    plt.show()


def demo02():
    # 创建一个空的无向图
    G = nx.Graph()

    # 添加节点
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)

    # 添加边，同时为每条边指定一个权重
    G.add_edge(1, 2, weight=0.5)
    G.add_edge(2, 3, weight=0.75)
    G.add_edge(1, 3, weight=0.6)

    # 为每条边指定一个颜色，这里我们使用权重来决定颜色
    # 权重越大，颜色越接近红色；权重越小，颜色越接近蓝色
    edge_colors = ['red' if G[u][v]['weight'] > 0.6 else 'blue' for u, v in G.edges()]

    # 绘制图形，使用 edge_color 参数指定边的颜色
    nx.draw(G, with_labels=True, edge_color=edge_colors, width=2, node_color='lightblue', node_size=700, alpha=0.7)

    # 显示图形
    plt.show()


def demo03():
    # 创建一个图
    G = nx.Graph()
    # 添加一些节点和边
    G.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 4), (4, 5), (5, 3)])

    # 为每个节点指定位置
    pos = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1), 5: (0.5, 2)}

    # 为每条边指定颜色
    edge_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']

    # 绘制图形，使用 pos 参数指定节点位置，使用 edgecolors 参数指定边的颜色
    nx.draw(G, pos, edge_color=edge_colors, with_labels=True, node_color='lightblue', node_size=700, alpha=0.7)

    # 显示图形
    plt.show()


def draw_network_digraph(input_num, hidden_num, output_num):
    G = nx.DiGraph()
    
    for i in range(input_num):
        for j in range (hidden_num):
            G.add_edge(i, input_num + j, weight=np.random.rand())
    
    for i in range(hidden_num):
        for j in range(output_num):
            G.add_edge(input_num + i, input_num + hidden_num + j, weight=np.random.rand())
            
    pos = dict()
    
    for i in range(0, input_num):
        pos[i] = (0, i - input_num / 2)
    
    for i in range(0, hidden_num):
        hidden = i + input_num
        pos[hidden] = (1, i - hidden_num / 2)
    
    for i in range(0, output_num):
        output = i + input_num + hidden_num
        pos[output] = (2, i - output_num / 2)
    
    # edge_colors = ['red' if G[u][v]['weight'] > 0.5 else 'green' for u, v in G.edges()]
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    
    nx.draw(G,
        pos,
        # with_labels=True,
        node_color='green',
        edge_color=edge_colors,
        width=0.5,
        node_size=300,
        edge_cmap=plt.cm.viridis,
        alpha=0.3)
    plt.show()

        
if __name__ == '__main__':
    # demo03()

    draw_network_digraph(5, 20, 2)
    