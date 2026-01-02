# -*- coding: UTF-8 -*-
import argparse
import os
from ultralytics import YOLO
import torch

def run_detection(weights_path, source_path, image_size=1024):
    """
    运行模型推理并保存结果
    :param weights_path: 训练好的模型权重路径 (best.pt)
    :param source_path: 待检测的图片或视频文件夹路径
    :param image_size: 输入图像尺寸，建议与训练时保持一致 (1024)
    """
    # 检查权重文件是否存在
    if not os.path.exists(weights_path):
        print(f"❌ 错误: 找不到权重文件 {weights_path}，请确认路径是否正确。")
        return

    # 1. 加载训练好的模型
    print(f"正在加载模型: {weights_path}...")
    model = YOLO(weights_path)
    
    # 2. 执行推理
    # conf: 置信度阈值，低于此值的框将被过滤
    # iou: NMS 阈值，用于处理重叠框
    results = model.predict(
        source=source_path,
        imgsz=image_size,
        save=True,           # 自动保存检测后的图片/视频到 runs/detect 目录
        show=False,          # 如果在本地带显示的电脑上可以设为 True 实时观看
        conf=0.25,           # 默认置信度 0.25
        iou=0.7,             # 默认 IOU 0.7
        device='0' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\n✅ 检测完成！结果已保存在: {results[0].save_dir}")

if __name__ == '__main__':
    # 使用 argparse 方便从命令行调用
    parser = argparse.ArgumentParser(description="YOLOv11 施工安全检测推理脚本")
    
    # 参数 1: 模型权重路径
    parser.add_argument('--weights', type=str, default='weights/best.pt', 
                        help='训练好的 best.pt 路径')
    
    # 参数 2: 待检测源 (可以是文件夹、单张图片或视频文件)
    parser.add_argument('--source', type=str, default='data/test_images', 
                        help='待检测的图片、文件夹或视频路径')
    
    # 参数 3: 图像尺寸
    parser.add_argument('--imgsz', type=int, default=1024, 
                        help='输入图像的分辨率')

    args = parser.parse_args()
    
    run_detection(args.weights, args.source, args.imgsz)