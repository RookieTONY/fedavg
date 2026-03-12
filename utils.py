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


def plot_comparison_results(
    results: Dict[str, Dict],
    save_path: str,
    experiment_name: str = "压缩算法对比实验"
):
    """
    绘制对比实验结果的综合图表
    包含：准确率曲线、通信成本对比、训练时长对比

    Args:
        results: 实验结果字典，格式为 {
            "none": {"history": {...}, "analysis": {...}},
            "topk": {"history": {...}, "analysis": {...}},
            "quantize": {"history": {...}, "analysis": {...}}
        }
        save_path: 保存路径
        experiment_name: 实验名称
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置中文字体（可选）
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 颜色方案 - 支持6组实验
    color_palette = [
        '#2ecc71',  # 绿色 - 无压缩
        '#3498db',  # 蓝色 - TopK 5%
        '#9b59b6',  # 紫色 - TopK 10%
        '#e67e22',  # 橙色 - TopK 20%
        '#e74c3c',  # 红色 - 量化 8bit
        '#1abc9c'   # 青色 - 量化 16bit
    ]

    # 动态分配颜色和标签
    result_keys = list(results.keys())
    colors = {key: color_palette[i % len(color_palette)] for i, key in enumerate(result_keys)}
    labels = {key: results[key].get('display_name', key) for key in result_keys}

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # ==================== 子图1: 准确率曲线对比 ====================
    ax1 = fig.add_subplot(gs[0, :])

    max_rounds = 0
    for comp_type, result in results.items():
        history = result['history']
        if history['test_accuracy']:
            rounds = range(1, len(history['test_accuracy']) + 1)
            ax1.plot(rounds, history['test_accuracy'],
                    marker='o', markersize=4, linewidth=2,
                    color=colors.get(comp_type, '#95a5a6'),
                    label=labels.get(comp_type, comp_type))
            max_rounds = max(max_rounds, len(history['test_accuracy']))

    ax1.set_xlabel('轮次 (Round)', fontsize=11)
    ax1.set_ylabel('测试准确率 (Test Accuracy)', fontsize=11)
    ax1.set_title('各压缩算法的测试准确率曲线', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0.5, max_rounds + 0.5)

    # ==================== 子图2: 最终准确率对比 ====================
    ax2 = fig.add_subplot(gs[1, 0])

    comp_types = list(results.keys())
    final_accs = []
    best_accs = []

    for comp_type in comp_types:
        analysis = results[comp_type]['analysis']
        final_accs.append(analysis.get('final_test_accuracy', 0))
        best_accs.append(analysis.get('best_test_accuracy', 0))

    x = np.arange(len(comp_types))
    width = 0.35

    bars1 = ax2.bar(x - width/2, final_accs, width, label='最终准确率',
                    color=[colors.get(ct, '#95a5a6') for ct in comp_types], alpha=0.8)
    bars2 = ax2.bar(x + width/2, best_accs, width, label='最佳准确率',
                    color=[colors.get(ct, '#95a5a6') for ct in comp_types], alpha=0.5)

    ax2.set_ylabel('准确率 (Accuracy)', fontsize=10)
    ax2.set_title('准确率对比', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels.get(ct, ct) for ct in comp_types], fontsize=8, rotation=15, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # ==================== 子图3: 总通信成本对比 ====================
    ax3 = fig.add_subplot(gs[1, 1])

    total_costs = []
    for comp_type in comp_types:
        analysis = results[comp_type]['analysis']
        total_costs.append(analysis.get('total_communication_cost', 0))

    bars = ax3.bar(x, total_costs,
                   color=[colors.get(ct, '#95a5a6') for ct in comp_types], alpha=0.8)

    ax3.set_ylabel('通信成本 (MB)', fontsize=10)
    ax3.set_title('总通信成本对比', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([labels.get(ct, ct) for ct in comp_types], fontsize=8, rotation=15, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    # 添加数值标签和节省百分比
    baseline_cost = total_costs[0] if total_costs else 0
    for i, (bar, cost) in enumerate(zip(bars, total_costs)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.2f}', ha='center', va='bottom', fontsize=9)

        if i > 0 and baseline_cost > 0:
            savings = (baseline_cost - cost) / baseline_cost * 100
            ax3.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                    f'节省\n{savings:.1f}%', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

    # ==================== 子图4: 平均每轮通信成本对比 ====================
    ax4 = fig.add_subplot(gs[1, 2])

    avg_costs = []
    for comp_type in comp_types:
        history = results[comp_type]['history']
        if history['communication_cost']:
            avg_costs.append(np.mean(history['communication_cost']))
        else:
            avg_costs.append(0)

    bars = ax4.bar(x, avg_costs,
                   color=[colors.get(ct, '#95a5a6') for ct in comp_types], alpha=0.8)

    ax4.set_ylabel('通信成本 (MB/轮)', fontsize=10)
    ax4.set_title('平均每轮通信成本', fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([labels.get(ct, ct) for ct in comp_types], fontsize=8, rotation=15, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, cost in zip(bars, avg_costs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.3f}', ha='center', va='bottom', fontsize=9)

    # ==================== 子图5: 通信成本累积曲线 ====================
    ax5 = fig.add_subplot(gs[2, :])

    for comp_type, result in results.items():
        history = result['history']
        if history['communication_cost']:
            rounds = range(1, len(history['communication_cost']) + 1)
            cumulative = np.cumsum(history['communication_cost'])
            ax5.plot(rounds, cumulative,
                    marker='o', markersize=4, linewidth=2,
                    color=colors.get(comp_type, '#95a5a6'),
                    label=labels.get(comp_type, comp_type))

    ax5.set_xlabel('轮次 (Round)', fontsize=11)
    ax5.set_ylabel('累积通信成本 (MB)', fontsize=11)
    ax5.set_title('累积通信成本曲线', fontsize=11, fontweight='bold')
    ax5.legend(loc='upper left', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # ==================== 总标题 ====================
    fig.suptitle(experiment_name, fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"对比实验结果图表已保存至: {save_path}")
    plt.close()


def plot_training_history_comparison(
    results: Dict[str, Dict],
    save_path: str,
    experiment_name: str = "训练历史对比"
):
    """
    绘制训练历史对比图表（更详细的版本）

    Args:
        results: 实验结果字典
        save_path: 保存路径
        experiment_name: 实验名称
    """
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 颜色方案 - 支持6组实验
    color_palette = [
        '#2ecc71',  # 绿色 - 无压缩
        '#3498db',  # 蓝色 - TopK 5%
        '#9b59b6',  # 紫色 - TopK 10%
        '#e67e22',  # 橙色 - TopK 20%
        '#e74c3c',  # 红色 - 量化 8bit
        '#1abc9c'   # 青色 - 量化 16bit
    ]

    # 动态分配颜色和标签
    result_keys = list(results.keys())
    colors = {key: color_palette[i % len(color_palette)] for i, key in enumerate(result_keys)}
    labels = {key: results[key].get('display_name', key) for key in result_keys}

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # 训练损失
    ax = axes[0, 0]
    for comp_type, result in results.items():
        history = result['history']
        if history['train_loss']:
            rounds = range(1, len(history['train_loss']) + 1)
            ax.plot(rounds, history['train_loss'],
                   marker='o', markersize=3, linewidth=2,
                   color=colors.get(comp_type, '#95a5a6'),
                   label=labels.get(comp_type, comp_type))
    ax.set_xlabel('轮次')
    ax.set_ylabel('训练损失')
    ax.set_title('训练损失对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 训练准确率
    ax = axes[0, 1]
    for comp_type, result in results.items():
        history = result['history']
        if history['train_accuracy']:
            rounds = range(1, len(history['train_accuracy']) + 1)
            ax.plot(rounds, history['train_accuracy'],
                   marker='o', markersize=3, linewidth=2,
                   color=colors.get(comp_type, '#95a5a6'),
                   label=labels.get(comp_type, comp_type))
    ax.set_xlabel('轮次')
    ax.set_ylabel('训练准确率')
    ax.set_title('训练准确率对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 测试损失
    ax = axes[1, 0]
    for comp_type, result in results.items():
        history = result['history']
        if history['test_loss']:
            rounds = range(1, len(history['test_loss']) + 1)
            ax.plot(rounds, history['test_loss'],
                   marker='o', markersize=3, linewidth=2,
                   color=colors.get(comp_type, '#95a5a6'),
                   label=labels.get(comp_type, comp_type))
    ax.set_xlabel('轮次')
    ax.set_ylabel('测试损失')
    ax.set_title('测试损失对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 测试准确率
    ax = axes[1, 1]
    for comp_type, result in results.items():
        history = result['history']
        if history['test_accuracy']:
            rounds = range(1, len(history['test_accuracy']) + 1)
            ax.plot(rounds, history['test_accuracy'],
                   marker='o', markersize=3, linewidth=2,
                   color=colors.get(comp_type, '#95a5a6'),
                   label=labels.get(comp_type, comp_type))
    ax.set_xlabel('轮次')
    ax.set_ylabel('测试准确率')
    ax.set_title('测试准确率对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.suptitle(experiment_name, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练历史对比图表已保存至: {save_path}")
    plt.close()


def create_comparison_summary_table(
    results: Dict[str, Dict],
    save_path: str = None
) -> str:
    """
    创建对比实验结果摘要表格（文本格式）

    Args:
        results: 实验结果字典
        save_path: 保存路径（可选）

    Returns:
        表格字符串
    """
    from tabulate import tabulate

    table_data = []
    headers = ['压缩类型', '最终准确率', '最佳准确率', '总通信量(MB)', '平均每轮通信量(MB)', '收敛轮次']

    for comp_type, result in results.items():
        analysis = results[comp_type]['analysis']
        history = results[comp_type]['history']
        display_name = result.get('display_name', comp_type)

        avg_comm = (analysis['total_communication_cost'] / len(history['communication_cost'])
                   if history['communication_cost'] else 0)

        row = [
            display_name,
            f"{analysis.get('final_test_accuracy', 0):.4f}",
            f"{analysis.get('best_test_accuracy', 0):.4f}",
            f"{analysis.get('total_communication_cost', 0):.4f}",
            f"{avg_comm:.4f}",
            analysis.get('convergence_round', 0)
        ]
        table_data.append(row)

    table_str = tabulate(table_data, headers=headers, tablefmt='grid')

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(table_str)
        print(f"对比摘要表格已保存至: {save_path}")

    return table_str
