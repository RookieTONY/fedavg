"""
数据集处理文件
包含CIFAR-10数据集的加载和Non-IID数据划分
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict


class NonIIDSplit:
    """
    Non-IID数据划分类
    使用Dirichlet分布控制数据异构程度
    """

    def __init__(self, dataset, num_clients, alpha=0.5, seed=42):
        """
        Args:
            dataset: 原始数据集
            num_clients: 客户端数量
            alpha: Dirichlet分布参数，越小数据异构程度越高
            seed: 随机种子
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed
        self.client_indices = self._split_data()

    def _split_data(self):
        """划分数据"""
        np.random.seed(self.seed)

        # 获取标签
        if hasattr(self.dataset, 'targets'):
            labels = np.array(self.dataset.targets)
        else:
            labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])

        num_classes = len(np.unique(labels))
        client_indices = defaultdict(list)

        # 对每个类别进行划分
        for class_idx in range(num_classes):
            # 获取该类别的所有样本索引
            class_indices = np.where(labels == class_idx)[0]
            np.random.shuffle(class_indices)

            # 使用Dirichlet分布生成每个客户端的样本比例
            proportions = np.random.dirichlet(
                np.repeat(self.alpha, self.num_clients)
            )
            proportions = (proportions * len(class_indices)).astype(int)

            # 确保所有样本都被分配
            proportions[-1] = len(class_indices) - proportions[:-1].sum()

            # 分配索引
            start_idx = 0
            for client_id, proportion in enumerate(proportions):
                end_idx = start_idx + proportion
                client_indices[client_id].extend(
                    class_indices[start_idx:end_idx].tolist()
                )
                start_idx = end_idx

        return client_indices

    def get_client_dataset(self, client_id):
        """获取指定客户端的数据集"""
        return Subset(self.dataset, self.client_indices[client_id])


class CIFAR10Dataset(Dataset):
    """自定义CIFAR-10数据集类"""

    def __init__(self, root='./data', train=True, download=True, transform=None):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_transforms(train=True):
    """获取数据增强变换"""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])


def get_cifar10_data(batch_size=32, root='./data'):
    """获取CIFAR-10数据加载器"""

    # 训练集
    train_dataset = CIFAR10Dataset(
        root=root,
        train=True,
        transform=get_transforms(train=True)
    )

    # 测试集
    test_dataset = CIFAR10Dataset(
        root=root,
        train=False,
        transform=get_transforms(train=False)
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader, train_dataset, test_dataset


def get_client_data_loaders(dataset, client_indices, batch_size=32):
    """为客户端创建数据加载器"""

    client_dataset = Subset(dataset, client_indices)

    train_loader = DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    return train_loader, len(client_dataset)
