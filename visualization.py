import matplotlib
# 强制使用非交互式后端，防止在后台线程中创建窗口导致 macOS 崩溃
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
import os
import json


class ProgressTracker:
    """进度追踪器"""
    
    def __init__(self, num_clients: int, num_rounds: int):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        
        # 全局进度
        self.current_round = 0
        self.round_start_time = None
        self.total_start_time = time.time()
        
        # 客户端状态
        self.client_status = {i: 'idle' for i in range(num_clients)}
        self.client_progress = {i: 0.0 for i in range(num_clients)}
        self.client_loss = {i: 0.0 for i in range(num_clients)}
        self.client_acc = {i: 0.0 for i in range(num_clients)}
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'communication_cost': [],
            'round_times': []
        }
        
        self.lock = threading.Lock()
        
    def update_round(self, round_num: int):
        """更新轮次"""
        with self.lock:
            self.current_round = round_num
            if round_num == 1:
                self.total_start_time = time.time()
                
    def update_client_status(self, client_id: int, status: str, 
                            progress: float = 0.0, loss: float = 0.0, acc: float = 0.0):
        """更新客户端状态"""
        with self.lock:
            self.client_status[client_id] = status
            self.client_progress[client_id] = progress
            self.client_loss[client_id] = loss
            self.client_acc[client_id] = acc
            
    def add_history(self, train_loss: float, train_acc: float, 
                   test_loss: float, test_acc: float, comm_cost: float, round_time: float):
        """添加历史记录"""
        with self.lock:
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_accuracy'].append(test_acc)
            self.history['communication_cost'].append(comm_cost)
            self.history['round_times'].append(round_time)


class RealtimeVisualizer:
    """
    实时可视化器 (后台保存模式)
    由于服务器占用主线程，可视化器将图表保存为文件而不是显示窗口
    """
    
    def __init__(self, tracker: ProgressTracker, config):
        self.tracker = tracker
        self.config = config
        self.fig = None
        self.running = False
        
        # 保存路径
        self.save_dir = os.path.join(config.LOG_DIR, 'visualizations')
        os.makedirs(self.save_dir, exist_ok=True)
        self.snapshot_path = os.path.join(self.save_dir, 'latest_progress.png')
        
        print(f"可视化图片将保存至: {self.snapshot_path}")
        
        # 颜色方案
        self.colors = {
            'primary': '#3498db',
            'secondary': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#9b59b6'
        }
        
    def create_dashboard(self):
        """创建仪表盘"""
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle('联邦学习实时监控 (后台生成中)', fontsize=14, fontweight='bold')
        
        # 创建网格布局
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # 1. 总体进度
        self.ax_progress = self.fig.add_subplot(gs[0, :])
        
        # 2. 客户端状态
        self.ax_clients = []
        for i in range(min(8, self.tracker.num_clients)):
            row = i // 4
            col = i % 4
            ax = self.fig.add_subplot(gs[1, col] if row == 0 else gs[2, col])
            self.ax_clients.append(ax)
        
        # 3. 性能曲线
        self.ax_loss = self.fig.add_subplot(gs[1, 2:4])
        self.ax_acc = self.fig.add_subplot(gs[2, 2:4])
        
        plt.tight_layout()
        
    def update_dashboard(self):
        """更新仪表盘并保存图片"""
        try:
            if self.fig is None:
                self.create_dashboard()
                
            # 清除所有轴
            for ax in [self.ax_progress, self.ax_loss, self.ax_acc] + self.ax_clients:
                ax.clear()
                
            # 绘制总体进度
            self._draw_progress()
            
            # 绘制客户端状态
            for i, ax in enumerate(self.ax_clients):
                if i < self.tracker.num_clients:
                    self._draw_client_status(ax, i)
                else:
                    ax.axis('off')
                    
            # 绘制性能曲线
            self._draw_curves()
            
            # 保存到文件
            self.fig.savefig(self.snapshot_path, dpi=100, bbox_inches='tight')
            
        except Exception as e:
            # 避免绘图错误影响主训练流程
            pass
            
    def _draw_progress(self):
        """绘制总体进度"""
        progress = self.tracker.current_round / self.tracker.num_rounds if self.tracker.num_rounds > 0 else 0
        
        # 进度条
        self.ax_progress.barh(0, progress, height=0.6, color=self.colors['primary'], alpha=0.8)
        self.ax_progress.barh(0, 1, height=0.6, color='gray', alpha=0.2)
        
        # 文字
        self.ax_progress.text(progress/2, 0, f'{progress*100:.1f}%',
                            ha='center', va='center', fontsize=12, 
                            fontweight='bold', color='white')
        
        self.ax_progress.text(1.02, 0, f'Round {self.tracker.current_round}/{self.tracker.num_rounds}',
                            ha='left', va='center', fontsize=10)
        
        # 预计时间
        elapsed = time.time() - self.tracker.total_start_time
        if self.tracker.current_round > 0 and progress > 0:
            eta = elapsed / progress - elapsed
            eta_str = f'预计剩余: {self._format_time(eta)}'
        else:
            eta_str = '预计剩余: 计算中...'
            
        self.ax_progress.text(-0.02, 0, eta_str, ha='right', va='center', fontsize=9)
        
        self.ax_progress.set_xlim(-0.3, 1.3)
        self.ax_progress.set_ylim(-0.5, 0.5)
        self.ax_progress.axis('off')
        self.ax_progress.set_title('总体进度', fontsize=11, fontweight='bold')
        
    def _draw_client_status(self, ax, client_id: int):
        """绘制客户端状态"""
        status = self.tracker.client_status[client_id]
        progress = self.tracker.client_progress[client_id]
        loss = self.tracker.client_loss[client_id]
        acc = self.tracker.client_acc[client_id]
        
        # 状态颜色
        colors = {
            'idle': '#95a5a6',
            'training': '#3498db',
            'evaluating': '#f39c12',
            'completed': '#2ecc71'
        }
        color = colors.get(status, '#95a5a6')
        
        # 状态圆圈
        circle = plt.Circle((0.5, 0.7), 0.15, color=color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(0.5, 0.7, f'C{client_id}', ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')
        
        # 状态文本
        status_text = {
            'idle': '等待中',
            'training': '训练中',
            'evaluating': '评估中',
            'completed': '已完成'
        }
        ax.text(0.5, 0.45, status_text.get(status, '未知'),
               ha='center', va='center', fontsize=9)
        
        # 进度条
        ax.barh(0.2, progress, height=0.1, color=self.colors['primary'], alpha=0.8)
        ax.barh(0.2, 1, height=0.1, color='gray', alpha=0.2)
        ax.text(0.5, 0.05, f'{progress*100:.0f}%', ha='center', va='center', fontsize=8)
        
        # 性能指标
        if status in ['training', 'completed']:
            info_text = f'Loss: {loss:.4f}\nAcc: {acc:.4f}'
            ax.text(0.5, -0.1, info_text, ha='center', va='center', fontsize=7,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.2, 1)
        ax.axis('off')
        
    def _draw_curves(self):
        """绘制性能曲线"""
        history = self.tracker.history
        
        # 损失曲线
        if history['train_loss']:
            rounds = range(1, len(history['train_loss']) + 1)
            self.ax_loss.plot(rounds, history['train_loss'], 
                            color=self.colors['primary'], 
                            marker='o', markersize=3, 
                            linewidth=2, label='训练')
            if history['test_loss']:
                self.ax_loss.plot(rounds[:len(history['test_loss'])], 
                                history['test_loss'],
                                color=self.colors['danger'], 
                                marker='s', markersize=3, 
                                linewidth=2, label='测试')
                
            self.ax_loss.set_xlabel('轮次')
            self.ax_loss.set_ylabel('损失')
            self.ax_loss.set_title('损失曲线', fontsize=11, fontweight='bold')
            self.ax_loss.legend(loc='upper right', fontsize=8)
            self.ax_loss.grid(True, alpha=0.3)
            
        # 准确率曲线
        if history['train_accuracy']:
            rounds = range(1, len(history['train_accuracy']) + 1)
            self.ax_acc.plot(rounds, history['train_accuracy'], 
                           color=self.colors['secondary'], 
                           marker='o', markersize=3, 
                           linewidth=2, label='训练')
            if history['test_accuracy']:
                self.ax_acc.plot(rounds[:len(history['test_accuracy'])], 
                               history['test_accuracy'],
                               color=self.colors['info'], 
                               marker='s', markersize=3, 
                               linewidth=2, label='测试')
                
            # 最佳准确率线
            if history['test_accuracy']:
                best_acc = max(history['test_accuracy'])
                self.ax_acc.axhline(y=best_acc, color='red', linestyle='--', 
                                  alpha=0.5, linewidth=1,
                                  label=f'最佳: {best_acc:.4f}')
                
            self.ax_acc.set_xlabel('轮次')
            self.ax_acc.set_ylabel('准确率')
            self.ax_acc.set_title('准确率曲线', fontsize=11, fontweight='bold')
            self.ax_acc.legend(loc='lower right', fontsize=8)
            self.ax_acc.grid(True, alpha=0.3)
            self.ax_acc.set_ylim(0, 1)
            
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f'{int(seconds)}秒'
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f'{minutes}分{secs}秒'
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f'{hours}小时{minutes}分'
            
    def start(self):
        """启动可视化循环"""
        self.running = True
        # 初始化画布
        self.create_dashboard()
        # 启动后台更新线程
        def update_loop():
            while self.running:
                self.update_dashboard()
                time.sleep(2)
        
        self.thread = threading.Thread(target=update_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """停止可视化"""
        self.running = False
        if self.fig:
            plt.close(self.fig)


class ProgressReporter:
    """进度报告器 (终端输出)"""

    def __init__(self, tracker: ProgressTracker):
        self.tracker = tracker

    def print_progress(self):
        """打印进度"""
        progress = self.tracker.current_round / self.tracker.num_rounds if self.tracker.num_rounds > 0 else 0
        bar_length = 50
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)

        elapsed = time.time() - self.tracker.total_start_time
        if progress > 0:
            eta = elapsed / progress - elapsed
            eta_str = self._format_time(eta)
        else:
            eta_str = "计算中..."

        print(f'\r[{bar}] {progress*100:.1f}% | '
              f'Round {self.tracker.current_round}/{self.tracker.num_rounds} | '
              f'已用: {self._format_time(elapsed)} | 预计: {eta_str}', end='', flush=True)

    def print_round_summary(self, round_num: int, train_loss: float, train_acc: float,
                           test_loss: float, test_acc: float, comm_cost: float):
        """打印轮次摘要"""
        print(f'\n\n{"="*70}')
        print(f'Round {round_num} 完成')
        print(f'{"="*70}')
        print(f'训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')
        print(f'测试 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')
        print(f'通信成本: {comm_cost:.2f} MB')
        print(f'{"="*70}\n')

    def print_client_status(self):
        """打印客户端状态"""
        print(f'\n{"="*70}')
        print('客户端状态:')
        print(f'{"="*70}')
        print(f'{"ID":<5} {"状态":<12} {"进度":<8} {"损失":<10} {"准确率":<10}')
        print(f'{"-"*70}')

        for client_id in range(self.tracker.num_clients):
            status = self.tracker.client_status[client_id]
            progress = self.tracker.client_progress[client_id]
            loss = self.tracker.client_loss[client_id]
            acc = self.tracker.client_acc[client_id]

            status_map = {
                'idle': '等待中',
                'training': '训练中',
                'evaluating': '评估中',
                'completed': '已完成'
            }

            print(f'{client_id:<5} {status_map.get(status, status):<12} '
                  f'{progress*100:>6.1f}%  {loss:>8.4f}  {acc:>8.4f}')

        print(f'{"="*70}\n')

    def print_final_summary(self):
        """打印最终摘要"""
        print(f'\n\n{"="*70}')
        print('实验完成!')
        print(f'{"="*70}')

        total_time = time.time() - self.tracker.total_start_time
        print(f'总用时: {self._format_time(total_time)}')
        print(f'完成轮次: {self.tracker.current_round}/{self.tracker.num_rounds}')

        if self.tracker.history['test_accuracy']:
            best_acc = max(self.tracker.history['test_accuracy'])
            best_round = self.tracker.history['test_accuracy'].index(best_acc) + 1
            print(f'最佳测试准确率: {best_acc:.4f} (第 {best_round} 轮)')

        if self.tracker.history['communication_cost']:
            total_cost = sum(self.tracker.history['communication_cost'])
            print(f'总通信成本: {total_cost:.2f} MB')

        print(f'{"="*70}\n')

    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f'{int(seconds)}秒'
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f'{minutes}分{secs}秒'
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f'{hours}小时{minutes}分'


# 全局实例
_progress_tracker: Optional[ProgressTracker] = None
_visualizer: Optional[RealtimeVisualizer] = None
_reporter: Optional[ProgressReporter] = None


def init_visualization(config):
    """初始化可视化系统"""
    global _progress_tracker, _visualizer, _reporter

    _progress_tracker = ProgressTracker(config.NUM_CLIENTS, config.NUM_ROUNDS)
    _visualizer = RealtimeVisualizer(_progress_tracker, config)
    _reporter = ProgressReporter(_progress_tracker)

    print("\n" + "="*70)
    print("可视化系统已初始化 (后台生成图片模式)")
    print("="*70)

    return _progress_tracker, _visualizer, _reporter


def get_tracker() -> Optional[ProgressTracker]:
    """获取进度追踪器"""
    return _progress_tracker


def get_visualizer() -> Optional[RealtimeVisualizer]:
    """获取可视化器"""
    return _visualizer


def get_reporter() -> Optional[ProgressReporter]:
    """获取报告器"""
    return _reporter


def update_progress(round_num: int = None, client_id: int = None, 
                   status: str = None, progress: float = None,
                   loss: float = None, acc: float = None):
    """更新进度的便捷函数"""
    global _progress_tracker, _visualizer, _reporter

    if _progress_tracker is None:
        return

    # 更新轮次
    if round_num is not None:
        _progress_tracker.update_round(round_num)

    # 更新客户端状态
    if client_id is not None:
        _progress_tracker.update_client_status(
            client_id, status or 'idle', progress or 0.0, loss or 0.0, acc or 0.0
        )

    # 打印进度
    if _reporter:
        _reporter.print_progress()