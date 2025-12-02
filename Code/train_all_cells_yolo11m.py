# 训练fulldatasets的yolo11m模型，支持13种类别的目标检测
from ultralytics import YOLO
import os
import time
import logging
import pandas as pd
from datetime import datetime
import yaml
# 导入路径配置文件
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paths_config import (
    PROJECT_ROOT,
    DATASETS_SMALL,
    DATASETS_FULL,
    MODELS_SMALL,
    MODELS_FULL,
    YOLO11M_MODEL
)

# 设置日志记录
log_dir = os.path.dirname(os.path.abspath(__file__))
log_filename = os.path.join(log_dir, f"train_all_cells_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义所有13种类别的映射
CELL_CATEGORIES = {
    "basophil": ["basophil"],  # 1种
    "eosinophil": ["eosinophil"],  # 1种
    "erythroblast": ["erythroblast"],  # 1种
    "ig": ["ig_promyelocyte", "ig_myelocyte", "ig_metamyelocyte", "ig_band"],  # 4种
    "lymphocyte": ["lymphocyte"],  # 1种
    "monocyte": ["monocyte"],  # 1种
    "neutrophil": ["neutrophil_seg", "neutrophil_band", "neutrophil_metamyelocyte"],  # 3种
    "platelet": ["platelet"]  # 1种
}

# 获取所有类别及其索引
ALL_CLASSES = []
for cell_type, sub_types in CELL_CATEGORIES.items():
    ALL_CLASSES.extend(sub_types)

# 类别到索引的映射
CLASS_TO_INDEX = {class_name: idx for idx, class_name in enumerate(ALL_CLASSES)}
INDEX_TO_CLASS = {idx: class_name for idx, class_name in enumerate(ALL_CLASSES)}

logger.info(f"总共13种类别: {ALL_CLASSES}")
logger.info(f"类别数量: {len(ALL_CLASSES)}")

def get_relative_paths():
    """
    获取基于项目根目录的路径字典
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    logger.info(f"当前脚本目录: {current_dir}")
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    
    return {
        "base_dir": PROJECT_ROOT,
        "full_models_dir": current_dir,  # full_models目录
        "full_datasets_dir": DATASETS_FULL,
        "small_datasets_dir": DATASETS_SMALL
    }

def generate_unified_data_yaml(paths):
    """
    生成统一的数据配置文件，包含所有13种类别
    """
    yaml_content = {
        "path": os.path.abspath(DATASETS_FULL),  # 使用绝对路径确保正确
        "train": "train",  # 后续会创建统一的训练集目录
        "val": "val",  # 后续会创建统一的验证集目录
        "test": "test",  # 可选
        "nc": len(ALL_CLASSES),
        "names": ALL_CLASSES
    }
    
    # 保存到full_models目录下
    yaml_path = os.path.join(paths["full_models_dir"], "full_dataset_all_classes.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, allow_unicode=True)
    
    logger.info(f"生成统一数据配置文件: {yaml_path}")
    return yaml_path

def prepare_combined_dataset(paths):
    """
    准备组合数据集，将所有类别的图片和标签整合到统一目录
    """
    # 创建统一的训练和验证目录
    train_images_dir = os.path.join(paths["full_datasets_dir"], "train", "images")
    train_labels_dir = os.path.join(paths["full_datasets_dir"], "train", "labels")
    val_images_dir = os.path.join(paths["full_datasets_dir"], "val", "images")
    val_labels_dir = os.path.join(paths["full_datasets_dir"], "val", "labels")
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    logger.info(f"创建统一数据集目录结构")
    
    # 收集并处理所有数据
    total_images = 0
    total_labels = 0
    
    for cell_type, sub_types in CELL_CATEGORIES.items():
        # 原数据路径
        src_images_dir = os.path.join(paths["full_datasets_dir"], cell_type, "images")
        src_labels_dir = os.path.join(paths["full_datasets_dir"], cell_type, "labels")
        
        if not os.path.exists(src_images_dir) or not os.path.exists(src_labels_dir):
            logger.warning(f"跳过 {cell_type}: 源目录不存在")
            continue
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(src_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in image_files:
            img_name = os.path.splitext(img_file)[0]
            src_img_path = os.path.join(src_images_dir, img_file)
            src_label_path = os.path.join(src_labels_dir, f"{img_name}.txt")
            
            # 创建目标文件名，添加细胞类型前缀以避免冲突
            target_img_name = f"{cell_type}_{img_file}"
            target_label_name = f"{cell_type}_{img_name}.txt"
            
            # 90%用于训练，10%用于验证
            import random
            if random.random() < 0.9:
                dst_images_dir = train_images_dir
                dst_labels_dir = train_labels_dir
            else:
                dst_images_dir = val_images_dir
                dst_labels_dir = val_labels_dir
            
            # 复制图片文件
            import shutil
            dst_img_path = os.path.join(dst_images_dir, target_img_name)
            shutil.copy2(src_img_path, dst_img_path)
            total_images += 1
            
            # 处理标签文件，更新类别索引
            if os.path.exists(src_label_path):
                dst_label_path = os.path.join(dst_labels_dir, target_label_name)
                
                with open(src_label_path, 'r', encoding='utf-8') as src, \
                     open(dst_label_path, 'w', encoding='utf-8') as dst:
                    
                    for line in src:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # 获取原始类别索引
                            orig_cls_idx = int(parts[0])
                            
                            # 确定实际的类别名称
                            if len(sub_types) == 1:
                                # 单类别的情况
                                class_name = sub_types[0]
                            else:
                                # 多类别的情况，根据索引选择对应的子类
                                if orig_cls_idx < len(sub_types):
                                    class_name = sub_types[orig_cls_idx]
                                else:
                                    # 索引超出范围，跳过
                                    logger.warning(f"跳过无效的类别索引 {orig_cls_idx} 在文件 {src_label_path}")
                                    continue
                            
                            # 获取统一的类别索引
                            if class_name in CLASS_TO_INDEX:
                                new_cls_idx = CLASS_TO_INDEX[class_name]
                                # 写入更新后的标签
                                dst.write(f"{new_cls_idx} {' '.join(parts[1:])}\n")
                                total_labels += 1
                            else:
                                logger.warning(f"未找到类别 {class_name} 的索引映射")
    
    logger.info(f"准备完成，总共 {total_images} 张图片，{total_labels} 个标签")
    return True

def train_unified_model(paths, yaml_path):
    """
    训练统一的yolo11m模型
    """
    # 模型路径 - 先检查是否已有下载的yolo11m.pt
    model_path = YOLO11M_MODEL
    if not os.path.exists(model_path):
        logger.info(f"yolo11m.pt 不存在，将自动下载")
        model_path = "yolo11m.pt"  # 使用ultralytics内置的模型路径，会自动下载
    
    output_dir = os.path.join(paths["full_models_dir"], "all_cells_train")
    
    logger.info(f"模型路径: {model_path}")
    logger.info(f"数据配置: {yaml_path}")
    logger.info(f"输出目录: {output_dir}")
    
    # 清理旧的训练目录
    if os.path.exists(output_dir):
        import shutil
        logger.info(f"清理旧的训练目录: {output_dir}")
        shutil.rmtree(output_dir)
    
    try:
        # 加载模型
        model = YOLO(model_path)
        logger.info(f"成功加载yolo11m模型")
        
        # 开始训练
        logger.info(f"开始训练统一模型，包含13种类别")
        
        # 训练参数设置
        train_results = model.train(
            data=yaml_path,
            epochs=100,  # 可以根据需要调整
            imgsz=640,  # yolo11m推荐的输入尺寸
            batch=4,  # 根据GPU内存调整
            device=0,  # 使用第一张GPU
            workers=4,  # 工作线程数
            save=True,
            project=paths["full_models_dir"],
            name="all_cells_train",
            amp=True,  # 启用自动混合精度
            verbose=True,
            patience=20,  # 早停耐心值
            freeze=0,  # 不冻结层，从头训练
            # 数据增强参数
            augment=True,
            hsv_h=0.015,  # 色调变换
            hsv_s=0.7,    # 饱和度变换
            hsv_v=0.4,    # 亮度变换
            # 优化器设置
            optimizer="AdamW",
            lr0=0.001,    # 初始学习率
            lrf=0.01      # 最终学习率
        )
        
        logger.info(f"✅ 统一模型训练完成")
        return True, output_dir
        
    except Exception as e:
        logger.error(f"❌ 训练过程出错: {str(e)}")
        return False, None

def evaluate_model(output_dir):
    """
    评估训练完成的模型
    """
    if not output_dir:
        return False, None
    
    best_model_path = os.path.join(output_dir, "weights", "best.pt")
    results_csv = os.path.join(output_dir, "results.csv")
    
    logger.info(f"评估模型路径: {best_model_path}")
    logger.info(f"结果文件路径: {results_csv}")
    
    if not os.path.exists(best_model_path):
        logger.warning(f"⚠️ 最佳模型文件不存在")
        return False, None
    
    try:
        df = pd.read_csv(results_csv)
        epochs_completed = len(df)
        
        # 关键指标
        metrics = {
            "mAP50": df['metrics/mAP50(B)'].max() if 'metrics/mAP50(B)' in df.columns else 0.0,
            "mAP50-95": df['metrics/mAP50-95(B)'].max() if 'metrics/mAP50-95(B)' in df.columns else 0.0,
            "precision": df['metrics/precision(B)'].max() if 'metrics/precision(B)' in df.columns else 0.0,
            "recall": df['metrics/recall(B)'].max() if 'metrics/recall(B)' in df.columns else 0.0
        }
        
        logger.info(f"\n模型评估结果:")
        logger.info(f"  训练轮数: {epochs_completed}")
        logger.info(f"  最佳mAP50: {metrics['mAP50']:.4f}")
        logger.info(f"  最佳mAP50-95: {metrics['mAP50-95']:.4f}")
        logger.info(f"  最佳精确率: {metrics['precision']:.4f}")
        logger.info(f"  最佳召回率: {metrics['recall']:.4f}")
        logger.info(f"  模型路径: {best_model_path}")
        
        # 保存类别映射信息
        class_mapping_path = os.path.join(output_dir, "class_mapping.txt")
        with open(class_mapping_path, 'w', encoding='utf-8') as f:
            f.write("类别索引映射关系:\n")
            for idx, class_name in sorted(INDEX_TO_CLASS.items()):
                f.write(f"{idx}: {class_name}\n")
        
        logger.info(f"类别映射已保存至: {class_mapping_path}")
        
        return True, metrics
        
    except Exception as e:
        logger.error(f"评估模型出错: {str(e)}")
        return False, None

def main():
    """
    主函数：训练包含13种类别的统一yolo11m模型
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("开始训练fulldatasets的yolo11m模型，包含13种类别")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 获取相对路径
    paths = get_relative_paths()
    
    # 生成统一的数据配置文件
    yaml_path = generate_unified_data_yaml(paths)
    
    # 准备组合数据集
    logger.info("开始准备组合数据集...")
    if not prepare_combined_dataset(paths):
        logger.error("准备数据集失败")
        return
    
    # 训练统一模型
    logger.info("开始训练统一模型...")
    success, output_dir = train_unified_model(paths, yaml_path)
    
    if success:
        # 评估模型
        logger.info("开始评估模型...")
        eval_success, metrics = evaluate_model(output_dir)
    
    # 输出总结信息
    total_elapsed = time.time() - start_time
    h, m, s = int(total_elapsed / 3600), int((total_elapsed % 3600) / 60), int(total_elapsed % 60)
    
    logger.info("=" * 60)
    logger.info(f"训练完成! 总耗时: {h}小时 {m}分钟 {s}秒")
    logger.info("=" * 60)
    
    if success and eval_success:
        logger.info(f"训练结果保存在: {output_dir}")
        logger.info(f"最佳模型位置: {os.path.join(output_dir, 'weights', 'best.pt')}")
        logger.info(f"类别映射信息: {os.path.join(output_dir, 'class_mapping.txt')}")
    
    logger.info(f"日志已保存至: {log_filename}")

if __name__ == "__main__":
    main()