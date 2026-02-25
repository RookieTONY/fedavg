# 1.依赖文件
flwr>=1.0.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
tqdm>=4.65.0
tensorboard>=2.12.0

# 2.使用 
# 2.1 基本使用

## 安装依赖
pip install -r requirements.txt

## 运行单次实验
python run.py --experiment single --rounds 10 --clients 5

## 运行对比实验
python run.py --experiment comparison --rounds 10 --clients 5

## 使用Non-IID数据分布
python run.py --experiment single --non_iid

## 使用不同压缩算法
python run.py --compression topk --rounds 15
python run_.py --compression quantize --rounds 15
python run_.py --compression none --rounds 15

# 2.2 高级配置

## 自定义配置运行
from config import Config
from run import run_experiment

config = Config()
config.NUM_ROUNDS = 20
config.NUM_CLIENTS = 10
config.COMPRESSION_TYPE = "topk"
config.TOPK_RATIO = 0.2
config.NON_IID = True
config.DIRICHLET_ALPHA = 0.3

history, analysis = run_experiment(config)