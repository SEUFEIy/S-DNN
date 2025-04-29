import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt


def plot_linear():
    baseline = 0.535
    baseline_SRN = 0.538
    seq_SRN = [0.823, 0.664, 0.624, 0.540, 0.538]
    seq_baseline = [0.8190000653266907, 0.6547500491142273, 0.6114999651908875, 0.5058750510215759, 0.508899986743927]

    plt.figure()
    plt.title('Linear Cifar10 Increament 3072-1100-10')

    x_index = np.arange(1, 6)
    plt.plot(x_index, seq_SRN, marker='*', markersize=5, c='red', label='seq_SRN', linewidth=3)
    plt.plot(x_index, seq_baseline, marker='*', markersize=5, c='cyan', label='seq_baseline', linewidth=3)

    plt.hlines(y=baseline, xmin=1, xmax=5, colors='green', label='baseline', alpha=0.5, linewidth=3)
    plt.hlines(y=baseline_SRN, xmin=1, xmax=5, colors='blue', label='baseline_SRN', alpha=0.5, linewidth=3)

    plt.ylim([0.45, 0.9])
    plt.xlim([1, 5])
    plt.legend()
    plt.xlabel('Number of task')
    plt.ylabel('Test accuracy')
    plt.tight_layout()
    # plt.show()
    plt.savefig('t.png', dpi=300)


def plot_cnn():
    baseline = 0.783
    baseline_SRN = 0.786
    seq_SRN = [0.899, 0.827750027179718, 0.824499988079071, 0.7850000042915344, 0.7852000069618225]
    seq_baseline = [0.8255000114440918, 0.5797500014305115, 0.6154999732971191, 0.48512503504753113, 0.4790000021457672]

    plt.figure()
    plt.title('CNN Cifar10 Increament 3-35-35')

    x_index = np.arange(1, 6)
    plt.plot(x_index, seq_SRN, marker='*', markersize=5, c='red', label='seq_SRN', linewidth=3)
    plt.plot(x_index, seq_baseline, marker='*', markersize=5, c='cyan', label='seq_baseline', linewidth=3)

    plt.hlines(y=baseline, xmin=1, xmax=5, colors='green', label='baseline', alpha=0.5, linewidth=3)
    plt.hlines(y=baseline_SRN, xmin=1, xmax=5, colors='blue', label='baseline_SRN', alpha=0.5, linewidth=3)

    plt.ylim([0.45, 0.9])
    plt.xlim([1, 5])
    plt.legend()
    plt.xlabel('Number of task')
    plt.ylabel('Test accuracy')
    plt.tight_layout()
    # plt.show()
    plt.savefig('t.png', dpi=300)

plot_cnn()
