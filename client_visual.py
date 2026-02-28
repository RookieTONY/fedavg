"""
联邦学习客户端实现（带可视化）
基于Flower框架的NumPyClient
"""
import flwr as fl
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple, Optional

from model import get_model
from compression import get_compressor, calculate_communication_cost
from config import Config

# 尝试导入可视化模块
try:
    from visualization import update_progress, get_tracker
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class CIFARClient(fl.client.NumPyClient):
    """
    CIFAR-10联邦学习客户端
    实现本地训练和参数更新
    """

    def __init__(self, client_id: int, train_loader, test_loader, config: Config):
        """
        初始化客户端
        Args:
            client_id: 客户端ID
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            config: 配置对象
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        # 创建模型
        self.model = get_model(config.MODEL_NAME, config.NUM_CLASSES)
        self.model.to(config.DEVICE)

        # 创建优化器
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )

        # 创建压缩器
        self.compressor = get_compressor(
            config.COMPRESSION_TYPE,
            compression_ratio=config.TOPK_RATIO,
            bits=config.QUANTIZE_BITS
        )

        # 记录训练统计信息
        self.training_history = []
        self.communication_stats = {
            'upload_size': 0.0,
            'download_size': 0.0,
            'compression_ratio': 0.0
        }

        # 连接可视化追踪器
        if VISUALIZATION_AVAILABLE:
            tracker = get_tracker()
            if tracker:
                self.progress_tracker = tracker
                print(f"客户端 {client_id} 已连接可视化追踪器")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        获取模型参数
        Args:
            config: 服务器传递的配置
        Returns:
            模型参数列表
        """
        print(f"Client {self.client_id}: Getting parameters...")

        # 获取模型参数
        parameters = [val.cpu().numpy() for val in self.model.state_dict().values()]

        # 计算原始大小
        original_size = calculate_communication_cost(parameters)

        # 应用压缩
        if self.compressor is not None and self.config.COMPRESSION_TYPE != "none":
            # 对于Top-K压缩，我们返回压缩后的数据
            compressed_data, metadata = self.compressor.compress(parameters)

            # 将压缩数据转换为可序列化的格式
            serialized_data = self._serialize_compressed_data(compressed_data, metadata)

            # 计算压缩后大小
            compressed_size = calculate_communication_cost(serialized_data)

            # 更新统计信息
            self.communication_stats['upload_size'] = compressed_size
            self.communication_stats['compression_ratio'] = (
                                                                    original_size - compressed_size
                                                            ) / original_size if original_size > 0 else 0.0

            print(
                f"Client {self.client_id}: Upload size reduced from {original_size:.4f} MB to {compressed_size:.4f} MB")
            print(f"Client {self.client_id}: Compression ratio: {self.communication_stats['compression_ratio']:.2%}")

            return serialized_data
        else:
            self.communication_stats['upload_size'] = original_size
            print(f"Client {self.client_id}: Upload size: {original_size:.4f} MB (no compression)")
            return parameters

    def _serialize_compressed_data(self, compressed_data: List, metadata: Dict) -> List[np.ndarray]:
        """
        序列化压缩数据
        将压缩数据转换为Flower可以传输的格式
        """
        serialized = []

        # 1. 添加基本元数据
        # [层数, 压缩类型标识, 维度信息总长度]
        num_layers = len(compressed_data)
        compression_type_byte = self.config.COMPRESSION_TYPE.encode('utf-8')[0]

        metadata_array = np.array([num_layers, compression_type_byte], dtype=np.int32)
        serialized.append(metadata_array)

        # 2. 序列化形状信息（改进版：保存每个形状的维度）
        # 存储格式：[形状总数, shape1_len, shape1_vals, shape2_len, shape2_vals, ...]
        shapes_info = []
        shapes_info.append(len(metadata['shapes']))  # 形状数量

        for shape in metadata['shapes']:
            shapes_info.append(len(shape))  # 当前形状的维度数
            shapes_info.extend(shape)  # 形状的各个维度值

        shapes_array = np.array(shapes_info, dtype=np.int32)
        serialized.append(shapes_array)

        # 3. 添加压缩后的数据
        if self.config.COMPRESSION_TYPE == "topk":
            for indices, values in compressed_data:
                serialized.append(indices)
                serialized.append(values)
        elif self.config.COMPRESSION_TYPE == "quantize":
            # 关键修正：序列化 scales 信息
            # metadata['scales'] 是一个浮点数列表
            scales_array = np.array(metadata['scales'], dtype=np.float32)
            serialized.append(scales_array)

            # 添加量化后的数据
            serialized.extend(compressed_data)
        else:
            serialized.extend(compressed_data)

        return serialized

    def _deserialize_compressed_data(self, parameters: List[np.ndarray]) -> Tuple:
        """
        反序列化压缩数据（修正版，支持Quantize）
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

        # 初始化 metadata 字典
        metadata = {'shapes': shapes}

        # 提取压缩数据
        if compression_type == 't':  # topk
            compressed_data = []
            param_idx = 2  # 从第三个参数开始

            for i in range(num_layers):
                indices = parameters[param_idx]
                values = parameters[param_idx + 1]
                compressed_data.append((indices, values))
                param_idx += 2

        elif compression_type == 'q':  # quantize
            # 关键修正：提取 scales 信息
            # scales 在 shapes 之后 (index 2)
            scales = parameters[2]
            metadata['scales'] = scales

            # 提取量化数据 (从 index 3 开始)
            compressed_data = parameters[3:3 + num_layers]

        else:
            # 其他压缩类型
            compressed_data = parameters[2:2 + num_layers]

        return compressed_data, compression_type, metadata

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        设置模型参数（修正版）
        """
        # 计算下载大小
        download_size = calculate_communication_cost(parameters)
        self.communication_stats['download_size'] = download_size
        print(f"Client {self.client_id}: Download size: {download_size:.4f} MB")

        # 如果使用压缩，先解压
        if (self.compressor is not None and
                self.config.COMPRESSION_TYPE != "none" and
                len(parameters) > 0 and
                len(parameters[0].shape) == 1 and
                parameters[0].shape[0] == 2):  # 检查是否是压缩格式

            # 解压数据（现在返回 compressed_data, compression_type, metadata）
            compressed_data, compression_type, metadata = self._deserialize_compressed_data(parameters)

            # 补充 metadata（TopK 需要 original_sizes）
            if compression_type == 't':
                metadata['original_sizes'] = [int(np.prod(shape)) for shape in metadata['shapes']]

            # 解压数据
            parameters = self.compressor.decompress(compressed_data, metadata)

        # 设置模型参数
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
            self,
            parameters: List[np.ndarray],
            config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        本地训练
        Args:
            parameters: 服务器下发的模型参数
            config: 训练配置
        Returns:
            训练后的参数、样本数、训练信息
        """
        print(f"Client {self.client_id}: Starting local training...")

        # 设置参数
        self.set_parameters(parameters)

        # 开始训练
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        start_time = time.time()

        for epoch in range(self.config.LOCAL_EPOCHS):
            # 更新客户端状态为训练中
            if VISUALIZATION_AVAILABLE and hasattr(self, 'progress_tracker'):
                update_progress(
                    client_id=self.client_id,
                    status='training',
                    progress=epoch / self.config.LOCAL_EPOCHS,
                    loss=0.0,
                    acc=0.0
                )

            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.config.DEVICE), targets.to(self.config.DEVICE)

                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                # 统计信息
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                epoch_total += targets.size(0)
                epoch_correct += predicted.eq(targets).sum().item()

                # 定期更新进度（每10个批次）
                if VISUALIZATION_AVAILABLE and hasattr(self, 'progress_tracker'):
                    if (batch_idx + 1) % 10 == 0:
                        avg_loss = epoch_loss / (batch_idx + 1)
                        avg_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0

                        update_progress(
                            client_id=self.client_id,
                            status='training',
                            progress=(epoch + batch_idx/len(self.train_loader)) / self.config.LOCAL_EPOCHS,
                            loss=avg_loss,
                            acc=avg_acc
                        )

            # 每个epoch的统计
            epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            print(f"Client {self.client_id}, Epoch {epoch + 1}: "
                  f"Loss: {epoch_loss / len(self.train_loader):.4f}, "
                  f"Acc: {epoch_acc:.4f}")

        training_time = time.time() - start_time

        # 计算最终指标
        final_loss = epoch_loss / len(self.train_loader)
        final_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0

        # 更新客户端状态为完成
        if VISUALIZATION_AVAILABLE and hasattr(self, 'progress_tracker'):
            update_progress(
                client_id=self.client_id,
                status='completed',
                progress=1.0,
                loss=final_loss,
                acc=final_accuracy
            )

        # 获取更新后的参数
        updated_parameters = self.get_parameters(config)

        # 返回训练信息
        history = {
            'train_loss': final_loss,
            'train_accuracy': final_accuracy,
            'training_time': training_time,
            'num_samples': len(self.train_loader.dataset),
            'client_id': self.client_id
        }

        print(f"Client {self.client_id}: Training completed in {training_time:.2f} seconds")

        return updated_parameters, len(self.train_loader.dataset), history

    def evaluate(
            self,
            parameters: List[np.ndarray],
            config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        评估模型性能
        Args:
            parameters: 模型参数
            config: 评估配置
        Returns:
            损失、样本数、评估信息
        """
        # 更新客户端状态为评估中
        if VISUALIZATION_AVAILABLE and hasattr(self, 'progress_tracker'):
            update_progress(
                client_id=self.client_id,
                status='evaluating',
                progress=0.0
            )

        print(f"Client {self.client_id}: Starting evaluation...")

        # 设置参数
        self.set_parameters(parameters)

        # 评估模式
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.config.DEVICE), targets.to(self.config.DEVICE)

                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # 计算准确率
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = test_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0

        print(f"Client {self.client_id}: Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # 返回评估结果
        eval_results = {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'client_id': self.client_id
        }

        return float(avg_loss), total, eval_results


def create_client(client_id: int, train_loader, test_loader, config: Config):
    """工厂函数，创建客户端实例"""
    return CIFARClient(client_id, train_loader, test_loader, config)
