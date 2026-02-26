"""
模型定义文件
包含用于CIFAR-10分类的卷积神经网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFARNet(nn.Module):
    """
    适用于CIFAR-10的卷积神经网络
    包含2个卷积层和2个全连接层
    """

    def __init__(self, num_classes=10):
        super(CIFARNet, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 第一个卷积块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # 第二个卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # 第三个卷积块
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_weights(self):
        """获取模型权重"""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights):
        """设置模型权重"""
        state_dict = {}
        for i, (key, _) in enumerate(self.state_dict().items()):
            state_dict[key] = torch.tensor(weights[i])
        self.load_state_dict(state_dict, strict=True)


class CIFARNetSmall(nn.Module):
    """
    轻量版CIFARNet，用于快速实验
    """

    def __init__(self, num_classes=10):
        super(CIFARNetSmall, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(model_name="CIFARNet", num_classes=10):
    """工厂函数，创建模型实例"""
    if model_name == "CIFARNet":
        return CIFARNet(num_classes=num_classes)
    elif model_name == "CIFARNetSmall":
        return CIFARNetSmall(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
