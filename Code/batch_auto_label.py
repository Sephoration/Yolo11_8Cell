# 批量自动标注所有细胞类型的full_datasets数据集
from ultralytics import YOLO
import os
import time
import logging
from datetime import datetime

# 设置日志记录
log_dir = os.path.dirname(os.path.abspath(__file__))
log_filename = os.path.join(log_dir, f"auto_label_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

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

# 定义所有细胞类型
cell_types = [
    {"name": "basophil", "chinese_name": "嗜碱性粒细胞"},
    {"name": "eosinophil", "chinese_name": "嗜酸性粒细胞"},
    {"name": "erythroblast", "chinese_name": "幼红细胞"},
    {"name": "ig", "chinese_name": "中幼粒细胞"},
    {"name": "lymphocyte", "chinese_name": "淋巴细胞"},
    {"name": "monocyte", "chinese_name": "单核细胞"},
    {"name": "neutrophil", "chinese_name": "中性粒细胞"},
    {"name": "platelet", "chinese_name": "血小板"}
]

def get_relative_paths():
    """
    获取基于脚本位置的相对路径
    """
    # 获取当前脚本所在目录（small_models）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取YOLO_8Cell根目录（small_models的父目录）
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    print(f"当前脚本目录: {current_dir}")
    print(f"项目根目录: {base_dir}")
    
    return {
        "base_dir": base_dir,
        "small_models_dir": current_dir,  # small_models目录
        "full_datasets_dir": os.path.join(base_dir, "full_datasets"),
        "small_datasets_dir": os.path.join(base_dir, "small_datasets")
    }

def auto_annotate_cell_type(cell_type, paths, conf_threshold=0.5, class_indices=None):
    """
    自动标注指定细胞类型的数据集
    
    Args:
        cell_type: 细胞类型字典，包含name和chinese_name
        paths: 路径字典
        conf_threshold: 置信度阈值
        class_indices: 类别索引映射，默认为None（使用默认索引）
    
    Returns:
        dict: 标注结果统计
    """
    cell_name = cell_type["name"]
    cell_chinese = cell_type["chinese_name"]
    
    # 构建模型和数据集路径
    model_dir = os.path.join(paths["small_models_dir"], f"{cell_name}_train")
    model_path = os.path.join(model_dir, "weights", "best.pt")
    source_dir = os.path.join(paths["full_datasets_dir"], cell_name, "images")
    output_dir = os.path.join(paths["full_datasets_dir"], cell_name)
    
    # 检查必要的路径是否存在
    if not os.path.exists(model_path):
        logger.error(f"{cell_chinese} ({cell_name}) 模型文件不存在: {model_path}")
        return {"success": False, "cell_type": cell_name, "message": "模型文件不存在"}
    
    if not os.path.exists(source_dir):
        logger.error(f"{cell_chinese} ({cell_name}) 图片目录不存在: {source_dir}")
        return {"success": False, "cell_type": cell_name, "message": "图片目录不存在"}
    
    # 创建输出标签目录
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    logger.info(f"开始标注 {cell_chinese} ({cell_name}) 数据集...")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"图片目录: {source_dir}")
    logger.info(f"输出目录: {labels_dir}")
    
    # 加载模型
    try:
        model = YOLO(model_path)
        logger.info(f"成功加载模型: {model_path}")
    except Exception as e:
        logger.error(f"加载 {cell_chinese} ({cell_name}) 模型失败: {str(e)}")
        return {"success": False, "cell_type": cell_name, "message": f"加载模型失败: {str(e)}"}
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(source_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(source_dir, file))
    
    if not image_files:
        logger.warning(f"在 {source_dir} 中未找到图片文件")
        return {"success": True, "cell_type": cell_name, "total_images": 0, "total_detections": 0}
    
    logger.info(f"找到 {len(image_files)} 张图片")
    
    total_detections = 0
    processed_images = 0
    start_time = time.time()
    
    # 处理每张图片
    for i, image_path in enumerate(image_files):
        filename = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(labels_dir, f"{filename}.txt")
        
        try:
            # 进行预测
            results = model(image_path, conf=conf_threshold, verbose=False)
            
            detections = 0
            with open(label_path, 'w', encoding='utf-8') as f:
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            # YOLO格式: class x_center y_center width height
                            cls = int(box.cls[0])
                            # 如果提供了类别索引映射，则使用映射后的索引
                            if class_indices and cls in class_indices:
                                cls = class_indices[cls]
                            
                            xywhn = box.xywhn[0].tolist()
                            # 写入类别和四个位置坐标
                            f.write(f"{cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")
                            detections += 1
            
            total_detections += detections
            processed_images += 1
            
            # 每处理10张图片输出一次进度
            if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                elapsed = time.time() - start_time
                rate = processed_images / elapsed if elapsed > 0 else 0
                logger.info(f"处理进度: {i+1}/{len(image_files)} 张图片, 检测到 {total_detections} 个细胞, 处理速率: {rate:.2f} 张/秒")
        
        except Exception as e:
            logger.error(f"处理图片 {filename} 时出错: {str(e)}")
    
    # 记录完成信息
    elapsed = time.time() - start_time
    logger.info(f"完成 {cell_chinese} ({cell_name}) 标注！")
    logger.info(f"处理了 {processed_images} 张图片")
    logger.info(f"总共检测到 {total_detections} 个细胞")
    logger.info(f"总耗时: {elapsed:.2f} 秒")
    logger.info(f"平均处理速率: {processed_images/elapsed:.2f} 张/秒")
    logger.info("-" * 60)
    
    return {
        "success": True,
        "cell_type": cell_name,
        "total_images": processed_images,
        "total_detections": total_detections,
        "elapsed_time": elapsed
    }

def get_special_class_indices():
    """
    获取特殊细胞类型的类别索引映射
    对于多类别数据集，可能需要调整类别索引
    """
    return {
        # 这里可以根据需要添加特殊的类别索引映射
        # 例如: "ig": {0: 0, 1: 1, 2: 2, 3: 3}  # 保持原有索引
        # 如果需要调整类别索引，可以在这里设置
    }

def main():
    """
    主函数：批量标注所有细胞类型的数据集
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("开始批量自动标注所有细胞类型数据集")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 获取相对路径
    paths = get_relative_paths()
    logger.info(f"项目根目录: {paths['base_dir']}")
    
    # 获取特殊类别索引映射
    special_indices = get_special_class_indices()
    
    # 标注统计
    results_summary = []
    success_count = 0
    total_processed_images = 0
    total_processed_detections = 0
    
    # 依次处理每种细胞类型
    for cell_type in cell_types:
        cell_name = cell_type["name"]
        # 获取该细胞类型的类别索引映射
        class_indices = special_indices.get(cell_name)
        
        result = auto_annotate_cell_type(cell_type, paths, conf_threshold=0.5, class_indices=class_indices)
        results_summary.append(result)
        
        if result["success"]:
            success_count += 1
            if "total_images" in result:
                total_processed_images += result["total_images"]
            if "total_detections" in result:
                total_processed_detections += result["total_detections"]
    
    # 输出汇总信息
    total_elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("批量标注完成！")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总耗时: {total_elapsed:.2f} 秒")
    logger.info(f"成功标注: {success_count}/{len(cell_types)} 种细胞类型")
    logger.info(f"总共处理: {total_processed_images} 张图片")
    logger.info(f"总共检测: {total_processed_detections} 个细胞")
    logger.info("\n详细结果:")
    
    for result in results_summary:
        if result["success"]:
            logger.info(f"- {result['cell_type']}: 处理 {result['total_images']} 张图片, 检测 {result['total_detections']} 个细胞, 耗时 {result['elapsed_time']:.2f} 秒")
        else:
            logger.error(f"- {result['cell_type']}: 失败 - {result['message']}")
    
    logger.info("=" * 60)
    logger.info(f"日志已保存至: {log_filename}")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()