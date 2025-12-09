# 训练13种类别细胞检测的YOLO11m模型
from ultralytics import YOLO        # 导入YOLO模型类
import os                           # 文件路径操作
import time                         # 计时和日志
import pandas as pd                 # 数据分析
import yaml                         # 配置文件生成
import shutil                       # 文件操作
import random                       # 数据划分
from pathlib import Path            # 路径操作

# ================================================================================
#                                   路径配置
# ================================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.absolute()
DATASETS_DIR = PROJECT_ROOT / "datasets_full"
MODELS_DIR = PROJECT_ROOT / "models_full"
PRETRAINED_MODEL = str(SCRIPT_DIR / "yolo11m.pt")  # 使用本地已存在的模型文件

# ================================================================================
#                                   类别配置（根据你的YAML文件）
# ================================================================================
ALL_CLASSES = [
    "basophil",
    "eosinophil", 
    "erythroblast",
    "ig_promyelocyte",
    "ig_myelocyte",
    "ig_metamyelocyte",
    "ig_band",
    "lymphocyte",
    "monocyte",
    "neutrophil_seg",
    "neutrophil_band",
    "neutrophil_metamyelocyte",
    "platelet"
]

# 映射：原始目录名 -> 最终类别名
DIR_TO_CLASSES = {
    "basophil": ["basophil"],
    "eosinophil": ["eosinophil"],
    "erythroblast": ["erythroblast"],
    "ig": ["ig", "ig_MMY", "ig_PMY", "ig_MY"],
    "lymphocyte": ["lymphocyte"],
    "monocyte": ["monocyte"],
    "neutrophil": ["BNE", "neutrophil", "SNE"],
    "platelet": ["platelet"]
}

CLASS_TO_INDEX = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}
INDEX_TO_CLASS = {idx: cls for idx, cls in enumerate(ALL_CLASSES)}

# ================================================================================
#                                   训练配置
# ================================================================================
TRAIN_CONFIG = {
    # 基础配置
    "epochs": 60,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "workers": 8,
    "amp": True,
    "verbose": True,
    "patience": 15,
    "save_best": True,
    "exist_ok": True,
    
    # 数据增强
    "augment": True,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    
    # 优化器
    "optimizer": "AdamW",
    "lr0": 0.01,
    "lrf": 0.1,
    "cos_lr": True,
    
    # 高级配置
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "close_mosaic": 5,
}

# ================================================================================
#                                   辅助函数
# ================================================================================
def get_class_index_from_label_file(label_path, cell_type):
    """从标签文件读取原始类别索引，转换为全局索引"""
    if not label_path.exists():
        return None
    
    with open(label_path, 'r') as f:
        first_line = f.readline().strip()
        if not first_line:
            return None
        
        parts = first_line.split()
        if len(parts) < 5:
            return None
        
        orig_idx = int(parts[0])
        target_classes = DIR_TO_CLASSES.get(cell_type, [])
        
        # 如果目录下只有一个类别，直接返回全局索引
        if len(target_classes) == 1:
            return CLASS_TO_INDEX.get(target_classes[0])
        
        # 如果目录下有多个类别，根据原始索引选择
        if 0 <= orig_idx < len(target_classes):
            target_class = target_classes[orig_idx]
            return CLASS_TO_INDEX.get(target_class)
        
    return None

def process_label_file(src_label_path, dst_label_path, global_class_idx):
    """处理标签文件，更新类别索引"""
    with open(src_label_path, 'r') as src, open(dst_label_path, 'w') as dst:
        for line in src:
            parts = line.strip().split()
            if len(parts) >= 5:
                # 替换为全局类别索引
                dst.write(f"{global_class_idx} {' '.join(parts[1:])}\n")

# ================================================================================
#                                   数据集准备函数
# ================================================================================
def prepare_balanced_dataset():
    """
    准备数据集，按类别分别划分训练集和验证集
    每个类别单独划分，确保类别平衡
    """
    combined_dir = DATASETS_DIR / "combined"
    
    # 创建目录结构
    (combined_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (combined_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (combined_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (combined_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)
    
    # 设置随机种子确保可复现
    random.seed(42)
    
    total_stats = {"train": 0, "val": 0, "by_class": {}}
    
    # 处理每个原始目录
    for cell_type, target_classes in DIR_TO_CLASSES.items():
        src_images = DATASETS_DIR / cell_type / "images"
        src_labels = DATASETS_DIR / cell_type / "labels"
        
        if not src_images.exists():
            print(f"跳过: {cell_type} 目录不存在")
            continue
        
        print(f"\n处理类别: {cell_type}")
        print(f"对应子类: {target_classes}")
        
        # 收集所有有效图片
        valid_images = []
        for img_path in src_images.glob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            img_name = img_path.stem
            label_path = src_labels / f"{img_name}.txt"
            
            # 确定图片的全局类别索引
            global_idx = get_class_index_from_label_file(label_path, cell_type)
            if global_idx is not None:
                valid_images.append((img_path, label_path, global_idx))
        
        print(f"找到 {len(valid_images)} 张有效图片")
        
        # 按类别分组
        images_by_class = {}
        for img_path, label_path, global_idx in valid_images:
            class_name = INDEX_TO_CLASS[global_idx]
            if class_name not in images_by_class:
                images_by_class[class_name] = []
            images_by_class[class_name].append((img_path, label_path, global_idx))
        
        # 按类别分别划分
        for class_name, image_list in images_by_class.items():
            if class_name not in total_stats["by_class"]:
                total_stats["by_class"][class_name] = {"train": 0, "val": 0}
            
            # 打乱当前类别的图片
            random.shuffle(image_list)
            
            # 90%训练，10%验证
            split_idx = int(len(image_list) * 0.9)
            train_list = image_list[:split_idx]
            val_list = image_list[split_idx:] if split_idx < len(image_list) else []
            
            # 处理训练集
            for img_path, label_path, global_idx in train_list:
                target_name = f"{cell_type}_{img_path.name}"
                
                # 复制图片
                shutil.copy2(img_path, combined_dir / "train" / "images" / target_name)
                
                # 处理标签
                if label_path.exists():
                    label_name = f"{cell_type}_{img_path.stem}.txt"
                    dst_label = combined_dir / "train" / "labels" / label_name
                    process_label_file(label_path, dst_label, global_idx)
                
                total_stats["train"] += 1
                total_stats["by_class"][class_name]["train"] += 1
            
            # 处理验证集
            for img_path, label_path, global_idx in val_list:
                target_name = f"{cell_type}_{img_path.name}"
                
                # 复制图片
                shutil.copy2(img_path, combined_dir / "val" / "images" / target_name)
                
                # 处理标签
                if label_path.exists():
                    label_name = f"{cell_type}_{img_path.stem}.txt"
                    dst_label = combined_dir / "val" / "labels" / label_name
                    process_label_file(label_path, dst_label, global_idx)
                
                total_stats["val"] += 1
                total_stats["by_class"][class_name]["val"] += 1
            
            print(f"  {class_name}: {len(train_list)} 训练, {len(val_list)} 验证")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("数据集统计:")
    print(f"训练集总计: {total_stats['train']} 张")
    print(f"验证集总计: {total_stats['val']} 张")
    print(f"总计: {total_stats['train'] + total_stats['val']} 张")
    print("\n按类别统计:")
    for class_name, counts in total_stats["by_class"].items():
        total = counts["train"] + counts["val"]
        train_ratio = counts["train"] / total if total > 0 else 0
        print(f"  {class_name}: {total} 张 (训练: {counts['train']}, 验证: {counts['val']}, 训练比例: {train_ratio:.1%})")
    print("="*60)
    
    return combined_dir if total_stats['train'] + total_stats['val'] > 0 else None

def generate_unified_data_yaml(combined_dir):
    """生成数据配置文件"""
    MODELS_DIR.mkdir(exist_ok=True)
    yaml_path = MODELS_DIR / "full_dataset_all_classes.yaml"
    
    yaml_content = {
        "path": str(combined_dir),
        "train": "train/images",
        "val": "val/images",
        "nc": len(ALL_CLASSES),
        "names": ALL_CLASSES
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, allow_unicode=True)
    
    print(f"生成YAML配置文件: {yaml_path}")
    return str(yaml_path)

# ================================================================================
#                                   训练和评估函数
# ================================================================================
def train_unified_model(yaml_path):
    """训练模型"""
    output_dir = MODELS_DIR / "all_cells_train"
    
    # 清理旧目录
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # 加载并训练模型
    model = YOLO(PRETRAINED_MODEL)
    train_config = TRAIN_CONFIG.copy()
    train_config["data"] = yaml_path
    train_config["project"] = str(MODELS_DIR)
    train_config["name"] = "all_cells_train"
    train_config["save"] = True
    
    print(f"\n开始训练，输出目录: {output_dir}")
    model.train(**train_config)
    
    print("模型训练完成")
    return output_dir

def evaluate_model(output_dir):
    """评估模型性能"""
    results_csv = output_dir / "results.csv"
    
    if not results_csv.exists():
        print("未找到结果文件")
        return None
    
    try:
        df = pd.read_csv(results_csv)
        
        metrics = {
            "epochs": len(df),
            "mAP50": df['metrics/mAP50(B)'].max() if 'metrics/mAP50(B)' in df.columns else 0.0,
            "mAP50-95": df['metrics/mAP50-95(B)'].max() if 'metrics/mAP50-95(B)' in df.columns else 0.0,
            "precision": df['metrics/precision(B)'].max() if 'metrics/precision(B)' in df.columns else 0.0,
            "recall": df['metrics/recall(B)'].max() if 'metrics/recall(B)' in df.columns else 0.0,
        }
        
        print(f"\n训练结果:")
        print(f"  训练轮数: {metrics['epochs']}")
        print(f"  最佳mAP50: {metrics['mAP50']:.4f}")
        print(f"  最佳mAP50-95: {metrics['mAP50-95']:.4f}")
        print(f"  最佳精确率: {metrics['precision']:.4f}")
        print(f"  最佳召回率: {metrics['recall']:.4f}")
        
        # 保存类别映射
        mapping_file = output_dir / "class_mapping.txt"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            for idx, name in INDEX_TO_CLASS.items():
                f.write(f"{idx}: {name}\n")
        
        print(f"类别映射保存至: {mapping_file}")
        return metrics
        
    except Exception as e:
        print(f"评估出错: {str(e)}")
        return None

# ================================================================================
#                                   主函数
# ================================================================================
def main():
    """主函数"""
    start_time = time.time()
    print("开始训练13种类别细胞检测模型")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查数据集目录
    if not DATASETS_DIR.exists():
        print(f"错误: 数据集目录不存在: {DATASETS_DIR}")
        return
    
    # 1. 准备数据集（按类别分别划分）
    print("\n步骤1: 准备数据集（按类别分别划分）...")
    combined_dir = prepare_balanced_dataset()
    if combined_dir is None:
        print("数据集准备失败")
        return
    
    # 2. 生成YAML配置文件
    print("\n步骤2: 生成数据配置文件...")
    yaml_path = generate_unified_data_yaml(combined_dir)
    
    # 3. 训练模型
    print("\n步骤3: 开始训练模型...")
    output_dir = train_unified_model(yaml_path)
    
    # 4. 评估模型
    print("\n步骤4: 评估模型...")
    evaluate_model(output_dir)
    
    # 总结
    elapsed = time.time() - start_time
    h, m, s = int(elapsed/3600), int(elapsed%3600/60), int(elapsed%60)
    
    print(f"\n训练完成! 总耗时: {h:02d}:{m:02d}:{s:02d}")
    print(f"模型保存在: {output_dir}")

if __name__ == "__main__":
    main()