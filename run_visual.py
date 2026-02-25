"""
联邦学习实验运行脚本（带可视化）
整合所有组件并运行实验
"""
import os
import sys
import argparse
import time
import threading
import multiprocessing
from typing import List, Dict
import flwr as fl

# 导入项目模块
from config import Config
from model import get_model
from dataset import get_cifar10_data, NonIIDSplit, get_client_data_loaders
from client_visual import create_client
from server_visual import start_server, create_strategy
from utils import (
    set_seed, get_experiment_id, save_experiment_config,
    plot_training_history, plot_communication_cost, analyze_results,
    print_results_summary
)

# 尝试导入可视化模块
try:
    from visualization import (
        init_visualization, get_visualizer, get_reporter, get_tracker
    )

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("警告: 可视化模块不可用，将使用标准模式运行")


def run_client(client_id: int, train_loader, test_loader, config: Config):
    """运行客户端"""
    print(f"Starting client {client_id}...")
    client = create_client(client_id, train_loader, test_loader, config)

    # 启动Flower客户端
    fl.client.start_client(
        server_address=config.SERVER_ADDRESS,
        client=client.to_client()
    )


def prepare_client_data(config: Config) -> List:
    """准备客户端数据"""
    print("Preparing client data...")

    # 获取完整数据集
    train_loader, test_loader, train_dataset, test_dataset = get_cifar10_data(
        batch_size=config.BATCH_SIZE,
        root=config.DATA_ROOT
    )

    client_loaders = []

    if config.NON_IID:
        # Non-IID数据划分
        splitter = NonIIDSplit(
            dataset=train_dataset,
            num_clients=config.NUM_CLIENTS,
            alpha=config.DIRICHLET_ALPHA,
            seed=config.SEED
        )

        for client_id in range(config.NUM_CLIENTS):
            client_dataset = splitter.get_client_dataset(client_id)
            client_loader, num_samples = get_client_data_loaders(
                train_dataset,
                client_dataset.indices,
                batch_size=config.BATCH_SIZE
            )
            client_loaders.append((client_loader, test_loader, num_samples))
            print(f"Client {client_id}: {num_samples} samples")
    else:
        # IID数据划分：每个客户端获得相同的数据
        for client_id in range(config.NUM_CLIENTS):
            client_loaders.append((train_loader, test_loader, len(train_dataset)))
            print(f"Client {client_id}: {len(train_dataset)} samples (IID)")

    return client_loaders


def launch_clients_in_background(config: Config, client_loaders: List):
    """
    在后台线程中启动客户端进程
    """
    print("\n客户端启动器已就绪，等待服务器启动...")
    time.sleep(5)  # 等待服务器完全启动

    print("开始启动客户端进程...")
    client_processes = []

    for client_id in range(min(config.NUM_CLIENTS, len(client_loaders))):
        train_loader, test_loader, num_samples = client_loaders[client_id]

        # 创建客户端进程
        process = multiprocessing.Process(
            target=run_client,
            args=(client_id, train_loader, test_loader, config)
        )
        process.start()
        client_processes.append(process)
        time.sleep(2)  # 间隔启动，避免同时连接

    # 等待所有客户端进程结束
    for process in client_processes:
        process.join()


def run_experiment(config: Config):
    """运行完整实验"""
    print("=" * 60)
    print("联邦学习实验开始")
    print("=" * 60)

    # 设置随机种子
    set_seed(config.SEED)

    # 生成实验ID
    experiment_id = get_experiment_id()
    experiment_dir = os.path.join(config.LOG_DIR, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)

    # 保存实验配置
    config_path = os.path.join(experiment_dir, "config.json")
    save_experiment_config(config, config_path)

    # 初始化可视化系统
    if VISUALIZATION_AVAILABLE:
        print("\n初始化可视化系统...")
        tracker, visualizer, reporter = init_visualization(config)

        # 注意：由于服务器占用主线程，可视化窗口需要以非阻塞方式运行
        # 这里我们主要依赖终端输出(ProgressReporter)来查看进度
        # 可视化窗口将在后台尝试更新，或者您可以查看保存的图表
        if visualizer:
            # 在后台线程启动可视化更新循环
            viz_thread = threading.Thread(target=visualizer.start, daemon=True)
            viz_thread.start()
            print("可视化仪表盘已在后台启动 (主要进度请查看终端输出)")
    else:
        tracker = None
        visualizer = None
        reporter = None

    # 准备客户端数据
    client_loaders = prepare_client_data(config)

    # ==========================================
    # 关键修正：在主线程运行服务器，在子线程启动客户端
    # ==========================================

    # 1. 启动一个后台线程，负责延迟启动所有客户端
    client_launcher_thread = threading.Thread(
        target=launch_clients_in_background,
        args=(config, client_loaders),
        daemon=True
    )
    client_launcher_thread.start()

    # 2. 在主线程运行服务器 (这会阻塞直到训练结束)
    try:
        print("\n正在主线程启动服务器...")
        history = start_server(config)
    except KeyboardInterrupt:
        print("\n\n用户中断实验...")
        history = None
    finally:
        # 清理可视化
        if VISUALIZATION_AVAILABLE and visualizer:
            print("\n停止可视化系统...")
            visualizer.stop()

    # 实验结束后的处理
    if history:
        # 分析结果
        analysis = analyze_results(history)
        print_results_summary(analysis)

        # 绘制图表
        plot_training_history(
            history,
            os.path.join(experiment_dir, "training_history.png")
        )

        plot_communication_cost(
            history,
            os.path.join(experiment_dir, "communication_cost.png")
        )

        # 保存分析结果
        import json
        analysis_path = os.path.join(experiment_dir, "analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"\n实验结果已保存到: {experiment_dir}")

    print("=" * 60)
    print("联邦学习实验完成")
    print("=" * 60)

    return history, {} if not history else analysis


def run_comparison_experiment(base_config: Config):
    """运行对比实验：不同压缩算法"""
    print("=" * 60)
    print("运行对比实验：不同压缩算法")
    print("=" * 60)

    compression_types = ["none", "topk", "quantize"]
    results = {}

    for compression_type in compression_types:
        print(f"\n运行实验: {compression_type} 压缩")

        # 创建配置副本
        config = Config()
        config.COMPRESSION_TYPE = compression_type

        if compression_type == "topk":
            config.TOPK_RATIO = 0.1
        elif compression_type == "quantize":
            config.QUANTIZE_BITS = 16

        # 运行实验
        history, analysis = run_experiment(config)
        results[compression_type] = {
            'history': history,
            'analysis': analysis
        }

    # 生成对比报告
    print("\n" + "=" * 60)
    print("对比实验结果")
    print("=" * 60)

    for compression_type, result in results.items():
        analysis = result['analysis']
        print(f"\n{compression_type.upper()} 压缩:")
        print(f"  最终测试准确率: {analysis['final_test_accuracy']:.4f}")
        print(f"  最佳测试准确率: {analysis['best_test_accuracy']:.4f}")
        print(f"  总通信成本: {analysis['total_communication_cost']:.4f} MB")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="联邦学习实验（带可视化）")
    parser.add_argument("--experiment", type=str, default="single",
                        choices=["single", "comparison"],
                        help="实验类型：single(单次实验) 或 comparison(对比实验)")
    parser.add_argument("--rounds", type=int, default=10,
                        help="联邦学习轮数")
    parser.add_argument("--clients", type=int, default=5,
                        help="客户端数量")
    parser.add_argument("--compression", type=str, default="topk",
                        choices=["none", "topk", "quantize"],
                        help="压缩算法")
    parser.add_argument("--non_iid", action="store_true",
                        help="使用Non-IID数据分布")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--local_epochs", type=int, default=2,
                        help="本地训练轮数")
    parser.add_argument("--no_viz", action="store_true",
                        help="禁用可视化")

    args = parser.parse_args()

    # 更新配置
    config = Config()
    config.NUM_ROUNDS = args.rounds
    config.NUM_CLIENTS = args.clients
    config.COMPRESSION_TYPE = args.compression
    config.NON_IID = args.non_iid
    config.BATCH_SIZE = args.batch_size
    config.LOCAL_EPOCHS = args.local_epochs

    # 检查是否禁用可视化
    if args.no_viz:
        global VISUALIZATION_AVAILABLE
        VISUALIZATION_AVAILABLE = False
        print("可视化已禁用")

    # 显示配置
    config.display()

    # 运行实验
    if args.experiment == "single":
        run_experiment(config)
    elif args.experiment == "comparison":
        run_comparison_experiment(config)


if __name__ == "__main__":
    main()