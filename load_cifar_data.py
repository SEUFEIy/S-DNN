import warnings
warnings.filterwarnings('ignore')

from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CustomCIFARDataset(Dataset):
    def __init__(self, dataset, target_classes, transform=None):
        if hasattr(dataset, 'dataset'):
            self.data = dataset.dataset.data[dataset.indices]
            self.targets = [dataset.dataset.targets[i] for i in dataset.indices]
        else:
            self.data = dataset.data
            self.targets = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
            
        self.transform = transform
        self.indices = [i for i, label in enumerate(self.targets) if label in target_classes]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = self.data[self.indices[idx]]
        label = self.targets[self.indices[idx]]

        if self.transform is not None:
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
                
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = datasets.CIFAR10(root='D:/Dataset/public/cifar10', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='D:/Dataset/public/cifar10', train=False, download=True, transform=transform)

    data = trainset.data
    labels = trainset.targets

    target_class = [0, 5, 3]
    custom_dataset = CustomCIFARDataset(trainset, target_class, transform=transform)
    dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

    for images, labels in dataloader:
        print(images.shape, labels.shape)
        break
