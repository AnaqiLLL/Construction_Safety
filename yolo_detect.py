# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-main 
@File    ：yolo_detect.py
@IDE     ：PyCharm 
@Author  ：Meng
@Date    ：2025/4/16 上午10:46 
@sesc    : 
'''
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载训练好的模型
    model = YOLO('runs/train/yolo11_safety/weights/best.pt')  # 替换为你的最佳模型路径
    metrics = model.val(data='data/safety.yaml', imgsz=640)
    # 图像推理
    results = model.val(
        source='data/11.mp4',  # 可以是文件/文件夹/URL/glob/PIL/numpy/mp4/
        # conf=0.25,  # 对象置信度阈值
        # iou=0.7,  # NMS IoU阈值
        imgsz=1440,  # 推理图像大小
        device='0',  # 使用GPU (可以是 '0', '0,1,2,3' 或 'cpu')
        show=False,  # 显示结果
        save=True,  # 保存结果
        save_txt=False,  # 保存结果为.txt文件
        save_conf=False,  # 保存结果带置信度
        save_crop=False,  # 保存裁剪的预测框
        show_labels=True,  # 显示标签
        show_conf=True,  # 显示置信度
        max_det=300,  # 每张图像最大检测数
        augment=False,  # 增强推理
        visualize=False,  # 可视化模型特征
        agnostic_nms=False,  # 类别无关NMS
        retina_masks=False,  # 使用高分辨率分割掩码
        classes=None,  # 按类别过滤结果
        boxes=True,  # 在分割预测中显示框
        line_thickness=3,  # 边界框厚度 (像素)
        half=False,  # 使用FP16半精度推理
        dnn=False,  # 使用OpenCV DNN进行ONNX推理
        vid_stride=1,  # 视频帧率步长
        stream_buffer=False,  # 缓冲所有流帧 (True) 或返回最新帧 (False)
        project='runs/detect',  # 保存项目名称
        name='exp',  # 实验名称
        exist_ok=False,  # 是否覆盖现有项目
    )

