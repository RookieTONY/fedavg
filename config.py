"""
联邦学习项目配置文件
包含所有实验参数和设置
"""
import torch
import os


class Config:
    """项目配置类"""

    # 数据集配置
    DATASET_NAME = "CIFAR10"
    DATA_ROOT = "./data"
    BATCH_SIZE = 32
    NUM_CLIENTS = 5  # 客户端数量
    NUM_CLASSES = 10

    # 模型配置
    MODEL_NAME = "CIFARNet"

    # 训练配置
    LOCAL_EPOCHS = 2  # 本地训练轮数
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

    # 联邦学习配置
    NUM_ROUNDS = 10  # 联邦学习轮数
    FRACTION_FIT = 1.0  # 每轮参与训练的客户端比例
    MIN_FIT_CLIENTS = 2  # 最少参与训练的客户端数量
    MIN_AVAILABLE_CLIENTS = 2  # 最少可用客户端数量

    # 压缩配置
    COMPRESSION_TYPE = "topk"  # 压缩类型: "topk", "quantize", "none"
    TOPK_RATIO = 0.1  # Top-K 压缩比例
    QUANTIZE_BITS = 16  # 量化位数 (16 或 8)

    # 实验配置
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 服务器配置
    SERVER_ADDRESS = "0.0.0.0:8080"

    # 数据分布配置 (Non-IID)
    NON_IID = True  # 是否使用Non-IID数据分布
    DIRICHLET_ALPHA = 0.5  # Dirichlet分布参数，控制数据异构程度

    # 日志配置
    LOG_DIR = "./logs"
    SAVE_MODEL = True
    MODEL_SAVE_PATH = "./saved_models"

    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        os.makedirs(cls.DATA_ROOT, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_SAVE_PATH, exist_ok=True)

    @classmethod
    def validate(cls):
        """验证配置参数"""
        assert cls.BATCH_SIZE > 0, "Batch size must be positive"
        assert cls.NUM_CLIENTS > 0, "Number of clients must be positive"
        assert cls.NUM_ROUNDS > 0, "Number of rounds must be positive"
        assert 0 < cls.TOPK_RATIO <= 1, "TopK ratio must be in (0, 1]"
        assert cls.QUANTIZE_BITS in [8, 16, 32], "Quantize bits must be 8, 16, or 32"

    @classmethod
    def display(cls):
        """显示配置信息"""
        print("=" * 50)
        print("联邦学习项目配置")
        print("=" * 50)
        for attr, value in cls.__dict__.items():
            if not attr.startswith('_') and not callable(value):
                print(f"{attr}: {value}")
        print("=" * 50)


# 初始化配置
Config.create_dirs()
Config.validate()
