# 批量自动标注所有细胞类型的full_datasets数据集
from ultralytics import YOLO
import os
import time
from pathlib import Path

# ================================================================================
#                                   路径配置
# ================================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.absolute()

DATASETS_FULL = PROJECT_ROOT / "datasets/datasets_full"
MODELS_SMALL = PROJECT_ROOT / "models/models_small"

# ================================================================================
#                                   标注配置
# ================================================================================
CELL_TYPES = [
    {"name": "basophil", "chinese_name": "嗜碱性粒细胞"},
    {"name": "eosinophil", "chinese_name": "嗜酸性粒细胞"},
    {"name": "erythroblast", "chinese_name": "幼红细胞"},
    {"name": "ig", "chinese_name": "中幼粒细胞"},
    {"name": "lymphocyte", "chinese_name": "淋巴细胞"},
    {"name": "monocyte", "chinese_name": "单核细胞"},
    {"name": "neutrophil", "chinese_name": "中性粒细胞"},
    {"name": "platelet", "chinese_name": "血小板"}
]

# 标注参数
CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值
BATCH_SIZE = 10  # 进度显示间隔

# ================================================================================
#                                   标注函数
# ================================================================================
def auto_annotate_cell_type(cell_type):
    """自动标注指定细胞类型的数据集"""
    cell_name = cell_type["name"]
    cell_chinese = cell_type["chinese_name"]
    
    # 构建路径
    model_path = MODELS_SMALL / f"{cell_name}_train" / "weights" / "best.pt"
    source_dir = DATASETS_FULL / cell_name / "images"
    labels_dir = DATASETS_FULL / cell_name / "labels"
    
    print(f"\n开始标注 {cell_chinese} ({cell_name})...")
    print(f"模型: {model_path}")
    print(f"图片源: {source_dir}")
    
    # 检查路径
    if not model_path.exists():
        print(f"错误: 模型文件不存在")
        return {"success": False, "cell_type": cell_name, "message": "模型文件不存在"}
    
    if not source_dir.exists():
        print(f"错误: 图片目录不存在")
        return {"success": False, "cell_type": cell_name, "message": "图片目录不存在"}
    
    # 创建标签目录
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    try:
        model = YOLO(str(model_path))
        print(f"成功加载模型")
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return {"success": False, "cell_type": cell_name, "message": f"加载模型失败: {str(e)}"}
    
    # 获取所有图片文件
    image_files = list(source_dir.glob("*.*"))
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in image_files if f.suffix.lower() in valid_extensions]
    
    if not image_files:
        print(f"警告: 未找到图片文件")
        return {"success": True, "cell_type": cell_name, "total_images": 0, "total_detections": 0}
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 开始标注
    total_detections = 0
    start_time = time.time()
    
    for i, image_path in enumerate(image_files):
        filename = image_path.stem
        label_path = labels_dir / f"{filename}.txt"
        
        try:
            # 进行预测
            results = model(str(image_path), conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            detections = 0
            with open(label_path, 'w', encoding='utf-8') as f:
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0])
                            xywhn = box.xywhn[0].tolist()
                            f.write(f"{cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")
                            detections += 1
            
            total_detections += detections
            
            # 显示进度
            if (i + 1) % BATCH_SIZE == 0 or (i + 1) == len(image_files):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"进度: {i+1}/{len(image_files)} 张, 检测: {total_detections} 个, 速率: {rate:.2f} 张/秒")
        
        except Exception as e:
            print(f"处理图片 {filename} 出错: {str(e)}")
    
    # 完成统计
    elapsed = time.time() - start_time
    print(f"完成标注 {cell_chinese}")
    print(f"处理图片: {len(image_files)} 张")
    print(f"检测目标: {total_detections} 个")
    print(f"总耗时: {elapsed:.2f} 秒")
    print(f"平均速率: {len(image_files)/elapsed:.2f} 张/秒")
    
    return {
        "success": True,
        "cell_type": cell_name,
        "total_images": len(image_files),
        "total_detections": total_detections,
        "elapsed_time": elapsed
    }

# ================================================================================
#                                   主函数
# ================================================================================
def main():
    """主函数：批量标注所有细胞类型的数据集"""
    print("开始批量自动标注所有细胞类型数据集")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查目录
    if not DATASETS_FULL.exists():
        print(f"错误: 数据集目录不存在: {DATASETS_FULL}")
        return
    
    if not MODELS_SMALL.exists():
        print(f"错误: 模型目录不存在: {MODELS_SMALL}")
        return
    
    # 标注统计
    results_summary = []
    success_count = 0
    total_images = 0
    total_detections = 0
    start_time = time.time()
    
    # 依次处理每种细胞类型
    for cell_type in CELL_TYPES:
        result = auto_annotate_cell_type(cell_type)
        results_summary.append(result)
        
        if result["success"]:
            success_count += 1
            total_images += result.get("total_images", 0)
            total_detections += result.get("total_detections", 0)
    
    # 输出汇总
    total_elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("批量标注完成!")
    print(f"总耗时: {total_elapsed:.2f} 秒")
    print(f"成功标注: {success_count}/{len(CELL_TYPES)} 种细胞类型")
    print(f"总共处理: {total_images} 张图片")
    print(f"总共检测: {total_detections} 个细胞")
    
    print(f"\n详细结果:")
    for result in results_summary:
        if result["success"]:
            print(f"  {result['cell_type']}: {result['total_images']} 张图片, {result['total_detections']} 个细胞")
        else:
            print(f"  {result['cell_type']}: 失败 - {result['message']}")
    
    print(f"\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == '__main__':
    main()