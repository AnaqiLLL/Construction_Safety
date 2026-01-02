# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-main 
@File    ：yolo_val.py
@IDE     ：PyCharm 
@Author  ：Meng
@Date    ：2025/4/16 上午10:45 
@sesc    : 
'''
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/yolo11_safety/weights/best.pt')
# source 改为你的测试图片路径
model.predict(source='data/construction_test.jpg', save=True, show=False)

if __name__ == '__main__':

    # 验证模型
    metrics = model.val(
        data='data/person/persion.yaml',  # 数据集配置文件路径
        batch=16,  # 批量大小
        imgsz=1440,  # 输入图像大小
        # conf=0.25,  # 对象置信度阈值
        # iou=0.6,  # NMS IoU阈值
        task='val',  # 可以是 'val', 'test' 或 'speed'
        device='0',  # 使用GPU (可以是 '0', '0,1,2,3' 或 'cpu')
        half=False,  # 使用FP16半精度推理
        dnn=False,  # 使用OpenCV DNN进行ONNX推理
        plots=True,  # 保存验证结果图
        save_json=False,  # 保存结果为JSON文件
        save_hybrid=False,  # 保存混合版本标签
        save_conf=False,  # 保存结果带置信度
        save_txt=False,  # 保存结果为.txt文件
        save_dir='runs/val',  # 保存目录
        name='exp',  # 实验名称
        exist_ok=False,  # 是否覆盖现有项目
        augment=False,  # 增强推理
        verbose=True,  # 打印详细输出
    )
