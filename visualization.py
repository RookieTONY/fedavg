import time
import threading
from datetime import datetime
from typing import Dict, List, Optional


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


class ProgressReporter:
    """进度报告器 (终端输出)"""

    def __init__(self, tracker: ProgressTracker):
        self.tracker = tracker
        self._last_print_time = 0
        self._last_round = -1

    def print_progress(self):
        """打印进度条 - 美化的命令行进度条"""
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

        # 获取最新的训练指标（如果有）
        info = ""
        if self.tracker.history['train_accuracy']:
            latest_acc = self.tracker.history['train_accuracy'][-1]
            info = f" | Acc: {latest_acc:.4f}"

        print(f'\r[{bar}] {progress*100:.1f}% | '
              f'Round {self.tracker.current_round}/{self.tracker.num_rounds}'
              f'{info} | '
              f'已用: {self._format_time(elapsed)} | 预计: {eta_str}', end='', flush=True)

    def print_round_summary(self, round_num: int, train_loss: float, train_acc: float,
                           test_loss: float, test_acc: float, comm_cost: float,
                           round_time: float = None):
        """打印轮次摘要 - 美化的输出格式"""
        print()
        print(f'{"="*70}')
        print(f'  Round {round_num}/{self.tracker.num_rounds} 完成')
        print(f'{"="*70}')
        print(f'  训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')
        print(f'  测试 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')
        print(f'  通信成本: {comm_cost:.4f} MB', end='')
        if round_time is not None:
            print(f', 轮次耗时: {round_time:.2f} 秒')
        else:
            print()
        print(f'{"="*70}')

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

        print(f'{"="*70}')

    def print_final_summary(self):
        """打印最终摘要 - 美化的输出格式"""
        print()
        print(f'{"="*70}')
        print('  实验完成!')
        print(f'{"="*70}')

        total_time = time.time() - self.tracker.total_start_time
        avg_round_time = total_time / self.tracker.num_rounds if self.tracker.num_rounds > 0 else 0
        print(f'  总用时: {self._format_time(total_time)} (平均每轮: {avg_round_time:.2f} 秒)')
        print(f'  完成轮次: {self.tracker.current_round}/{self.tracker.num_rounds}')

        if self.tracker.history['test_accuracy']:
            best_acc = max(self.tracker.history['test_accuracy'])
            best_round = self.tracker.history['test_accuracy'].index(best_acc) + 1
            final_acc = self.tracker.history['test_accuracy'][-1]
            print(f'  最佳测试准确率: {best_acc:.4f} (第 {best_round} 轮)')
            print(f'  最终测试准确率: {final_acc:.4f}')

        if self.tracker.history['communication_cost']:
            total_cost = sum(self.tracker.history['communication_cost'])
            avg_cost = total_cost / len(self.tracker.history['communication_cost'])
            print(f'  总通信成本: {total_cost:.4f} MB (平均每轮: {avg_cost:.4f} MB)')

        print(f'{"="*70}')
        print()

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
_reporter: Optional[ProgressReporter] = None


def init_visualization(config):
    """初始化可视化系统"""
    global _progress_tracker, _reporter

    _progress_tracker = ProgressTracker(config.NUM_CLIENTS, config.NUM_ROUNDS)
    _reporter = ProgressReporter(_progress_tracker)

    print("\n" + "="*70)
    print("进度追踪系统已初始化")
    print("="*70)

    return _progress_tracker, None, _reporter


def get_tracker() -> Optional[ProgressTracker]:
    """获取进度追踪器"""
    return _progress_tracker


def get_visualizer():
    """获取可视化器（已废弃，返回None）"""
    return None


def get_reporter() -> Optional[ProgressReporter]:
    """获取报告器"""
    return _reporter


def update_progress(round_num: int = None, client_id: int = None,
                   status: str = None, progress: float = None,
                   loss: float = None, acc: float = None):
    """更新进度的便捷函数"""
    global _progress_tracker, _reporter

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