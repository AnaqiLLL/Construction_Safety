# -*- coding: UTF-8 -*-

import warnings
import torch
import os

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # ================= 1. 环境准备 =================
    DEVICE = '0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ================= 2. 加载 V2 冠军权重 (41.4% 那个) =================
    # 我们回到 Nano 架构的巅峰状态进行“手术式”微调，确保起点最高
    model_path = r'runs/train/yolo11_safety_v2_balanced/weights/best.pt'
    if not os.path.exists(model_path):
        model_path = 'yolo11n.pt'

    model = YOLO(model_path)

    # ================= 3. V14 “精准手术”定向优化策略 =================
    # 策略核心：
    # - imgsz=1024: 识别 17 类中极小目标的物理前提。
    # - freeze=20: 【核心改进】冻结前 20 层。锁定 90% 的网络参数，
    #   确保挖掘机、装载机、自卸车等已经学好的知识“永不丢失”。
    # - cls=25.0: 【终极加压】25 倍分类损失增益。强迫仅存的可训练参数必须
    #   在手套、人员、安全鞋等“差生”类别上产生收敛。
    # - lr0=0.000001: 【微米级更新】使用 1e-6 的极低学习率，进行点穴式修正。

    model.train(
        data=r"data/safety.yaml",
        imgsz=1024,  # 必须 1024，否则小目标特征不足
        epochs=100,  # 手术级微调 100 轮
        patience=0,  # 关闭早停，确保每一轮的细微调整都生效
        batch=4,
        device=DEVICE,
        workers=2,
        optimizer='AdamW',
        lr0=0.000001,  # 【极低】确保不破坏原有高精度权重
        cos_lr=True,
        close_mosaic=50,  # 提前关闭增强，给小目标最稳定的环境
        amp=True,

        # --- V14 极端定向参数 ---
        freeze=20,  # 深度冻结，实现真正的“定向补强”
        box=5.0,  # 降低定位增益（大目标框已准，不需要大动）
        cls=25.0,  # 【暴击】25 倍分类增益，全力主攻 0 精度类别

        # --- 数据增强（强化小目标出镜率） ---
        copy_paste=0.9,  # 90% 概率强行塞入小目标，让模型必须看到它们
        scale=0.2,  # 锁定局部细节
        mixup=0.0,

        project='runs/train',
        name='yolo11_safety_v14_surgical_sniper',
        plots=True,
        save=True
    )

    print("\n✅ V14 定向手术级微调启动！")
    print("💡 逻辑：锁死了 20 层权重保分，利用 25 倍分类增益和超低学习率强攻手套和人员。")