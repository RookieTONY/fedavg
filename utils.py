"""
工具函数文件
包含各种辅助功能
"""
import torch
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any

from config import Config


def set_seed(seed: int = 42):
    """设置随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """加载模型检查点"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 训练损失
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 训练准确率
    axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 测试损失
    axes[1, 0].plot(history['test_loss'], label='Test Loss')
    axes[1, 0].set_title('Test Loss')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 测试准确率
    axes[1, 1].plot(history['test_accuracy'], label='Test Accuracy')
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_communication_cost(history: Dict[str, List[float]], save_path: str):
    """绘制通信成本曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['communication_cost'], label='Communication Cost (MB)')
    plt.title('Communication Cost per Round')
    plt.xlabel('Round')
    plt.ylabel('Cost (MB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Communication cost plot saved to {save_path}")
    plt.close()


def calculate_model_size(model):
    """计算模型大小（MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_experiment_id():
    """生成实验ID"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_experiment_config(config: Config, filepath: str):
    """保存实验配置"""
    config_dict = {
        key: value for key, value in config.__dict__.items()
        if not key.startswith('_') and not callable(value)
    }

    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"Experiment configuration saved to {filepath}")


def load_experiment_config(filepath: str) -> Dict:
    """加载实验配置"""
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config


class EarlyStopping:
    """早停类，用于防止过拟合"""

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class LearningRateScheduler:
    """学习率调度器"""

    def __init__(self, optimizer, mode='step', patience=3, factor=0.5):
        self.optimizer = optimizer
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.counter = 0
        self.best_loss = None

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0
        else:
            self.best_loss = val_loss
            self.counter = 0

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.factor
        print(f"Learning rate reduced to {self.optimizer.param_groups[0]['lr']}")


def analyze_results(history: Dict[str, List[float]]) -> Dict[str, Any]:
    """分析实验结果"""
    analysis = {
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
        'final_train_accuracy': history['train_accuracy'][-1] if history['train_accuracy'] else 0,
        'final_test_loss': history['test_loss'][-1] if history['test_loss'] else 0,
        'final_test_accuracy': history['test_accuracy'][-1] if history['test_accuracy'] else 0,
        'best_test_accuracy': max(history['test_accuracy']) if history['test_accuracy'] else 0,
        'total_communication_cost': sum(history['communication_cost']) if history['communication_cost'] else 0,
        'convergence_round': 0
    }

    # 计算收敛轮次
    if history['test_accuracy']:
        best_acc = max(history['test_accuracy'])
        analysis['convergence_round'] = history['test_accuracy'].index(best_acc) + 1

    return analysis


def print_results_summary(analysis: Dict[str, Any]):
    """打印结果摘要"""
    print("\n" + "=" * 50)
    print("实验结果摘要")
    print("=" * 50)
    print(f"最终训练损失: {analysis['final_train_loss']:.4f}")
    print(f"最终训练准确率: {analysis['final_train_accuracy']:.4f}")
    print(f"最终测试损失: {analysis['final_test_loss']:.4f}")
    print(f"最终测试准确率: {analysis['final_test_accuracy']:.4f}")
    print(f"最佳测试准确率: {analysis['best_test_accuracy']:.4f}")
    print(f"收敛轮次: {analysis['convergence_round']}")
    print(f"总通信成本: {analysis['total_communication_cost']:.4f} MB")
    print("=" * 50)
