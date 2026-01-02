from datasets import load_dataset
import os
from PIL import Image
from tqdm import tqdm

# 数据集配置
DATASET_ID = "keremberke/construction-safety-object-detection" 
SAVE_DIR = "data/construction_safety" 

# YOLO 格式转换函数：将 COCO [x_min, y_min, w, h] 转换为 YOLO [x_center, y_center, w_norm, h_norm]
def convert_to_yolo_bbox(bbox, width, height):
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / width
    y_center = (y_min + h / 2) / height
    w_norm = w / width
    h_norm = h / height
    return f"{x_center} {y_center} {w_norm} {h_norm}"

def process_split(dataset, split_name, class_names):
    # 确保目录结构符合 YOLO 要求
    images_dir = os.path.join(SAVE_DIR, split_name, "images")
    labels_dir = os.path.join(SAVE_DIR, split_name, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"Processing {split_name} data...")
    for item in tqdm(dataset):
        image = item['image']
        # 确保 image_id 是字符串，用于文件名
        image_id = str(item['image_id']) 
        objects = item['objects']
        
        img_width, img_height = image.size
        img_filename = f"{image_id}.jpg"
        
        # 1. 保存图片
        image.save(os.path.join(images_dir, img_filename))
        
        # 2. 保存 YOLO 标签 txt
        label_filename = f"{image_id}.txt"
        with open(os.path.join(labels_dir, label_filename), 'w') as f:
            for i in range(len(objects['id'])):
                # category 是类别索引
                cls_id = objects['category'][i] 
                bbox = objects['bbox'][i]
                
                # 类别索引必须是从 0 开始的整数
                yolo_box = convert_to_yolo_bbox(bbox, img_width, img_height)
                f.write(f"{cls_id} {yolo_box}\n")

# 3. 下载并处理数据
if __name__ == "__main__":
    # 下载数据
    ds = load_dataset(DATASET_ID, name="full")
    
    # 获取类别名称
    class_names = ds['train'].features['objects'].feature['category'].names
    print("Detected Class Names:", class_names)
    
    # 处理训练集和验证集
    if 'train' in ds: process_split(ds['train'], 'train', class_names)
    if 'validation' in ds: process_split(ds['validation'], 'val', class_names)
    if 'test' in ds: process_split(ds['test'], 'test', class_names)

    print(f"\n数据准备完成！保存在: {SAVE_DIR}")