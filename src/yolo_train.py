# -*- coding: UTF-8 -*-
from ultralytics import YOLO
import torch
import warnings

# 忽略不必要的库警告
warnings.filterwarnings('ignore')

def train_model():
    # 1. 硬件环境检查：优先使用 GPU (RTX 4060)
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 训练启动 | 使用设备: {device}")

    # 2. 加载基础模型权重 (YOLOv11 Nano)
    # 实验证明对于 300 张的小样本数据集，Nano 架构比 Small 架构更具泛化性
    model = YOLO('yolo11n.pt') 

    # 3. 开始训练 (执行 V2 平衡版调参策略)
    # 此配置在实验中达到了最高的 mAP50 (41.4%)
    model.train(
        data='data/safety.yaml',      # 数据集配置文件路径
        imgsz=1024,                   # 锁定 1024 高分辨率，这是捕捉手套等小目标的生命线
        epochs=300,                  # 充足的迭代轮数
        batch=16,                    # 4060 显存适配的最佳批次
        device=device,               # 指定训练设备
        optimizer='AdamW',           # 使用 AdamW 优化器处理复杂的 17 类分布
        lr0=0.001,                   # 初始学习率
        cos_lr=True,                 # 开启余弦退火学习率调度
        close_mosaic=20,             # 最后 20 轮关闭 Mosaic 增强以提高边界框精度
        
        # 结果保存路径
        project='runs/train',
        name='construction_safety_final',
        plots=True,                  # 生成结果图表供 Report 使用
        save=True
    )

if __name__ == '__main__':
    # 确保在 Windows 环境下正确运行多进程
    train_model()