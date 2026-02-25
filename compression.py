"""
压缩算法实现
包含Top-K、量化等多种压缩方法
"""
import numpy as np
import struct
from typing import List, Tuple


class GradientCompressor:
    """梯度压缩器基类"""

    def __init__(self, compression_ratio=0.1):
        self.compression_ratio = compression_ratio

    def compress(self, gradients: List[np.ndarray]) -> Tuple[List, dict]:
        """压缩梯度"""
        raise NotImplementedError

    def decompress(self, compressed_data: List, metadata: dict) -> List[np.ndarray]:
        """解压梯度"""
        raise NotImplementedError


class TopKCompressor(GradientCompressor):
    """
    Top-K压缩算法
    只保留绝对值最大的k%的梯度值
    """

    def __init__(self, compression_ratio=0.1):
        super().__init__(compression_ratio)

    def compress(self, gradients: List[np.ndarray]) -> Tuple[List, dict]:
        """
        压缩梯度
        Args:
            gradients: 梯度列表
        Returns:
            compressed_data: 压缩后的数据
            metadata: 元数据（包含形状、索引等）
        """
        compressed_data = []
        metadata = {'shapes': [], 'indices': [], 'original_sizes': []}

        for grad in gradients:
            flat_grad = grad.flatten()
            k = max(1, int(len(flat_grad) * self.compression_ratio))

            # 找到绝对值最大的k个元素的索引
            indices = np.argpartition(np.abs(flat_grad), -k)[-k:]

            # 获取对应的值
            values = flat_grad[indices]

            compressed_data.append((indices.astype(np.int32), values.astype(np.float32)))
            metadata['shapes'].append(grad.shape)
            metadata['original_sizes'].append(len(flat_grad))

        return compressed_data, metadata

    def decompress(self, compressed_data: List, metadata: dict) -> List[np.ndarray]:
        """
        解压梯度
        Args:
            compressed_data: 压缩后的数据
            metadata: 元数据
        Returns:
            gradients: 解压后的梯度列表
        """
        gradients = []

        for (indices, values), shape in zip(compressed_data, metadata['shapes']):
            # 创建全零数组
            flat_grad = np.zeros(metadata['original_sizes'][len(gradients)], dtype=np.float32)

            # 恢复非零值
            flat_grad[indices] = values

            # 重塑为原始形状
            gradients.append(flat_grad.reshape(shape))

        return gradients


class QuantizationCompressor(GradientCompressor):
    """
    量化压缩算法
    将float32量化为指定位数
    """

    def __init__(self, bits=16):
        super().__init__(1.0)  # 量化不改变数据量，只改变精度
        self.bits = bits

    def compress(self, gradients: List[np.ndarray]) -> Tuple[List, dict]:
        """量化压缩"""
        if self.bits == 16:
            dtype = np.float16
        elif self.bits == 8:
            dtype = np.int8
        else:
            dtype = np.float32

        compressed_data = []
        metadata = {'dtype': dtype, 'shapes': [], 'scales': []}

        for grad in gradients:
            if self.bits == 8:
                # 8位量化需要缩放因子
                scale = np.abs(grad).max() / 127.0 if np.abs(grad).max() > 0 else 1.0
                quantized = (grad / scale).astype(dtype)
                metadata['scales'].append(scale)
                compressed_data.append(quantized)
            else:
                compressed_data.append(grad.astype(dtype))
                metadata['scales'].append(1.0)

            metadata['shapes'].append(grad.shape)

        return compressed_data, metadata

    def decompress(self, compressed_data: List, metadata: dict) -> List[np.ndarray]:
        """反量化"""
        gradients = []

        for quantized, scale, shape in zip(
                compressed_data, metadata['scales'], metadata['shapes']
        ):
            if self.bits == 8:
                # 8位反量化
                grad = quantized.astype(np.float32) * scale
            else:
                grad = quantized.astype(np.float32)

            gradients.append(grad.reshape(shape))

        return gradients


class SparseCompressor(GradientCompressor):
    """
    稀疏压缩算法
    将小于阈值的梯度置零
    """

    def __init__(self, threshold=0.001):
        super().__init__(1.0)
        self.threshold = threshold

    def compress(self, gradients: List[np.ndarray]) -> Tuple[List, dict]:
        """稀疏压缩"""
        compressed_data = []
        metadata = {'shapes': []}

        for grad in gradients:
            # 创建掩码
            mask = np.abs(grad) >= self.threshold

            # 应用掩码
            sparse_grad = np.where(mask, grad, 0.0)
            compressed_data.append(sparse_grad)
            metadata['shapes'].append(grad.shape)

        return compressed_data, metadata

    def decompress(self, compressed_data: List, metadata: dict) -> List[np.ndarray]:
        """解压稀疏梯度"""
        return [grad.reshape(shape) for grad, shape in zip(compressed_data, metadata['shapes'])]


def get_compressor(compression_type="topk", **kwargs):
    """工厂函数，创建压缩器实例"""
    if compression_type == "topk":
        return TopKCompressor(compression_ratio=kwargs.get('compression_ratio', 0.1))
    elif compression_type == "quantize":
        return QuantizationCompressor(bits=kwargs.get('bits', 16))
    elif compression_type == "sparse":
        return SparseCompressor(threshold=kwargs.get('threshold', 0.001))
    elif compression_type == "none":
        return None
    else:
        raise ValueError(f"Unknown compression type: {compression_type}")


def calculate_compression_ratio(original_size, compressed_size):
    """计算压缩率"""
    if original_size == 0:
        return 0.0
    return (original_size - compressed_size) / original_size


def calculate_communication_cost(data_list):
    """计算通信成本（MB）"""
    total_bytes = 0

    for data in data_list:
        if isinstance(data, np.ndarray):
            total_bytes += data.nbytes
        elif isinstance(data, tuple):
            # 对于压缩数据
            indices, values = data
            total_bytes += indices.nbytes + values.nbytes

    return total_bytes / (1024 * 1024)  # 转换为MB
