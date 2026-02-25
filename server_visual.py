"""
联邦学习服务器端实现（带可视化）
"""
import flwr as fl
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
import time
from collections import defaultdict

from model import get_model
from compression import get_compressor, calculate_communication_cost
from config import Config

# 尝试导入可视化模块
try:
    from visualization import get_tracker, update_progress, get_reporter
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class FedAvgWithCompression(fl.server.strategy.FedAvg):
    """
    带压缩支持的FedAvg策略
    支持处理压缩后的客户端更新
    """

    def __init__(self, config: Config, *args, **kwargs):
        self.config = config
        self.server_model = get_model(config.MODEL_NAME, config.NUM_CLASSES)
        self.server_model.to(config.DEVICE)

        # 创建压缩器
        self.compressor = get_compressor(
            config.COMPRESSION_TYPE,
            compression_ratio=config.TOPK_RATIO,
            bits=config.QUANTIZE_BITS
        )

        # 记录训练历史
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'communication_cost': [],
            'round_times': []
        }

        # 轮次开始时间
        self.round_start_time = None

        # 连接可视化追踪器
        if VISUALIZATION_AVAILABLE:
            tracker = get_tracker()
            if tracker:
                self.tracker = tracker
                print("服务器已连接可视化追踪器")

        # 初始化父类
        super().__init__(*args, **kwargs)

    def _deserialize_compressed_data(self, parameters: List[np.ndarray]) -> Tuple:
        """
        反序列化压缩数据
        从客户端接收的压缩参数中重建原始参数
        """
        # 提取元数据
        metadata_array = parameters[0]
        num_layers = int(metadata_array[0])
        compression_type = chr(int(metadata_array[1]))

        # 提取并重建形状信息
        shapes_info = parameters[1]

        # 重建 shapes 列表
        shapes = []
        idx = 0
        num_shapes = int(shapes_info[idx])
        idx += 1

        for _ in range(num_shapes):
            shape_len = int(shapes_info[idx])
            idx += 1
            shape = tuple(int(shapes_info[idx + i]) for i in range(shape_len))
            shapes.append(shape)
            idx += shape_len

        # 提取压缩数据
        if compression_type == 't':  # topk
            compressed_data = []
            param_idx = 2  # 从第三个参数开始

            for i in range(num_layers):
                indices = parameters[param_idx]
                values = parameters[param_idx + 1]
                compressed_data.append((indices, values))
                param_idx += 2
        else:
            # 其他压缩类型
            compressed_data = parameters[2:2 + num_layers]

        return compressed_data, compression_type, shapes

    def _decompress_client_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """
        解压客户端参数
        将压缩格式转换为完整的模型参数
        """
        # 检查是否是压缩格式
        if (len(parameters) > 0 and
            len(parameters[0].shape) == 1 and
            parameters[0].shape[0] == 2):

            try:
                # 反序列化
                compressed_data, compression_type, shapes = self._deserialize_compressed_data(parameters)

                # 重建元数据
                metadata = {
                    'shapes': shapes,
                    'original_sizes': [int(np.prod(shape)) for shape in shapes]
                }

                # 解压
                decompressed = self.compressor.decompress(compressed_data, metadata)
                return decompressed
            except Exception as e:
                print(f"Warning: Failed to decompress parameters: {e}")
                return parameters
        else:
            return parameters

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Tuple[fl.server.client_proxy.ClientProxy, Exception]]
    ) -> Tuple[Optional[fl.common.Parameters], Dict]:
        """
        聚合客户端训练结果
        """
        print(f"\nServer: Aggregating results for round {server_round}")

        # 记录轮次开始时间
        self.round_start_time = time.time()

        if not results:
            return None, {}

        # 提取参数和样本数
        weights_results = []
        total_samples = 0
        round_stats = defaultdict(float)

        for client_proxy, fit_res in results:
            # 获取参数
            parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)

            # 获取样本数和训练信息
            num_examples = fit_res.num_examples
            metrics = fit_res.metrics if fit_res.metrics else {}

            # 计算通信成本
            comm_cost = calculate_communication_cost(parameters)
            round_stats['upload_cost'] += comm_cost

            # 累积统计信息
            if 'train_loss' in metrics:
                round_stats['total_loss'] += metrics['train_loss'] * num_examples
            if 'train_accuracy' in metrics:
                round_stats['total_accuracy'] += metrics['train_accuracy'] * num_examples
            if 'training_time' in metrics:
                round_stats['total_time'] += metrics['training_time']

            total_samples += num_examples

            # 关键修正：先解压再聚合
            decompressed_params = self._decompress_client_parameters(parameters)
            weights_results.append((decompressed_params, num_examples))

        # 计算平均统计信息
        avg_loss = round_stats['total_loss'] / total_samples if total_samples > 0 else 0.0
        avg_accuracy = round_stats['total_accuracy'] / total_samples if total_samples > 0 else 0.0

        print(f"Server: Round {server_round} - "
              f"Avg Loss: {avg_loss:.4f}, "
              f"Avg Accuracy: {avg_accuracy:.4f}, "
              f"Communication Cost: {round_stats['upload_cost']:.4f} MB")

        # 记录历史
        self.history['train_loss'].append(avg_loss)
        self.history['train_accuracy'].append(avg_accuracy)
        self.history['communication_cost'].append(round_stats['upload_cost'])

        # 计算轮次耗时
        round_time = time.time() - self.round_start_time if self.round_start_time else 0
        self.history['round_times'].append(round_time)

        # 聚合权重（现在已经解压，可以正常聚合）
        aggregated_parameters = self._aggregate_weights(weights_results)

        # 转换为Parameters对象
        aggregated_parameters_proto = fl.common.ndarrays_to_parameters(aggregated_parameters)

        # 更新可视化进度
        if VISUALIZATION_AVAILABLE and hasattr(self, 'tracker'):
            # 获取测试指标（如果有）
            test_loss = self.history['test_loss'][-1] if self.history['test_loss'] else 0.0
            test_acc = self.history['test_accuracy'][-1] if self.history['test_accuracy'] else 0.0

            # 更新进度追踪器
            self.tracker.add_history(
                avg_loss, avg_accuracy, test_loss, test_acc,
                round_stats['upload_cost'], round_time
            )

            # 更新可视化
            update_progress(round_num=server_round)

            # 打印轮次摘要
            reporter = get_reporter()
            if reporter:
                reporter.print_round_summary(
                    server_round, avg_loss, avg_accuracy,
                    test_loss, test_acc, round_stats['upload_cost']
                )

        # 返回聚合结果和统计信息
        metrics = {
            'avg_train_loss': avg_loss,
            'avg_train_accuracy': avg_accuracy,
            'total_samples': total_samples,
            'communication_cost': round_stats['upload_cost'],
            'num_clients': len(results)
        }

        return aggregated_parameters_proto, metrics

    def _aggregate_weights(self, weights_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        聚合权重
        Args:
            weights_results: [(权重, 样本数), ...]
        Returns:
            聚合后的权重
        """
        # 计算总样本数
        total_samples = sum(num_examples for _, num_examples in weights_results)

        # 初始化聚合权重
        aggregated_weights = None

        for weights, num_examples in weights_results:
            # 计算权重
            weight = num_examples / total_samples

            # 加权聚合
            if aggregated_weights is None:
                aggregated_weights = [w * weight for w in weights]
            else:
                for i, w in enumerate(weights):
                    aggregated_weights[i] += w * weight

        return aggregated_weights

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
            failures: List[Tuple[fl.server.client_proxy.ClientProxy, Exception]]
    ) -> Tuple[Optional[float], Dict]:
        """
        聚合评估结果
        """
        print(f"Server: Aggregating evaluation results for round {server_round}")

        if not results:
            return None, {}

        # 计算平均损失和准确率
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        for client_proxy, eval_res in results:
            num_examples = eval_res.num_examples
            loss = eval_res.loss
            metrics = eval_res.metrics if eval_res.metrics else {}

            total_loss += loss * num_examples
            total_accuracy += metrics.get('test_accuracy', 0.0) * num_examples
            total_samples += num_examples

        # 计算平均值
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0

        # 记录历史
        self.history['test_loss'].append(avg_loss)
        self.history['test_accuracy'].append(avg_accuracy)

        # 更新可视化追踪器中的测试指标
        if VISUALIZATION_AVAILABLE and hasattr(self, 'tracker'):
            # 更新最新的测试指标
            if self.tracker.history['train_loss']:
                # 确保测试指标与训练指标对齐
                if len(self.tracker.history['test_loss']) < len(self.tracker.history['train_loss']):
                    self.tracker.history['test_loss'].append(avg_loss)
                    self.tracker.history['test_accuracy'].append(avg_accuracy)

        print(f"Server: Round {server_round} Evaluation - "
              f"Avg Loss: {avg_loss:.4f}, "
              f"Avg Accuracy: {avg_accuracy:.4f}")

        metrics = {
            'avg_test_loss': avg_loss,
            'avg_test_accuracy': avg_accuracy,
            'total_samples': total_samples
        }

        return avg_loss, metrics

    def save_history(self, filepath: str):
        """保存训练历史"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Server: Training history saved to {filepath}")


class FedProxWithCompression(FedAvgWithCompression):
    """
    FedProx算法实现
    在FedAvg基础上添加近端项
    """

    def __init__(self, mu: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu  # 近端项系数

    def _aggregate_weights(self, weights_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """
        FedProx权重聚合
        """
        # FedProx的聚合方式与FedAvg相同，区别在于客户端训练时的目标函数
        # 客户端训练时会添加近端项，聚合方式不变
        return super()._aggregate_weights(weights_results)


class ScaffoldStrategy(FedAvgWithCompression):
    """
    SCAFFOLD算法实现
    通过控制变量减少客户端漂移
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化全局控制变量
        self.global_control = None

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Tuple[fl.server.client_proxy.ClientProxy, Exception]]
    ) -> Tuple[Optional[fl.common.Parameters], Dict]:
        """
        SCAFFOLD聚合
        """
        # 标准FedAvg聚合
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # 更新全局控制变量
        # 这里简化实现，实际SCAFFOLD需要维护和更新控制变量
        # 详情参考原始论文

        return aggregated_parameters, metrics


def create_strategy(config: Config):
    """工厂函数，创建聚合策略"""
    if config.COMPRESSION_TYPE == "fedprox":
        return FedProxWithCompression(
            config=config,
            fraction_fit=config.FRACTION_FIT,
            min_fit_clients=config.MIN_FIT_CLIENTS,
            min_available_clients=config.MIN_AVAILABLE_CLIENTS,
            mu=0.01
        )
    elif config.COMPRESSION_TYPE == "scaffold":
        return ScaffoldStrategy(
            config=config,
            fraction_fit=config.FRACTION_FIT,
            min_fit_clients=config.MIN_FIT_CLIENTS,
            min_available_clients=config.MIN_AVAILABLE_CLIENTS
        )
    else:
        return FedAvgWithCompression(
            config=config,
            fraction_fit=config.FRACTION_FIT,
            min_fit_clients=config.MIN_FIT_CLIENTS,
            min_available_clients=config.MIN_AVAILABLE_CLIENTS
        )


def start_server(config: Config):
    """启动联邦学习服务器"""
    print("=" * 50)
    print("Starting Federated Learning Server")
    print("=" * 50)
    print(f"Address: {config.SERVER_ADDRESS}")
    print(f"Number of rounds: {config.NUM_ROUNDS}")
    print(f"Strategy: {config.COMPRESSION_TYPE}")
    print("=" * 50)

    # 创建策略
    strategy = create_strategy(config)

    # 启动服务器
    fl.server.start_server(
        server_address=config.SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS),
        strategy=strategy,
    )

    # 保存训练历史
    history_path = f"{config.LOG_DIR}/training_history.json"
    strategy.save_history(history_path)

    # 打印最终摘要
    if VISUALIZATION_AVAILABLE and hasattr(strategy, 'tracker'):
        reporter = get_reporter()
        if reporter:
            reporter.print_final_summary()

    print("Server completed all rounds")
    return strategy.history
