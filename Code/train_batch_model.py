# 批量训练所有细胞类型的YOLO11n模型
from ultralytics import YOLO
import os
import time
import pandas as pd
import shutil
import random
from pathlib import Path

# ================================================================================
#                                   路径配置
# ================================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.absolute()

DATASETS_SMALL = PROJECT_ROOT / "datasets/datasets_small"
MODELS_SMALL = PROJECT_ROOT / "models/models_small"
PRETRAINED_MODEL = str(SCRIPT_DIR / "yolo11n.pt")  # 使用本地已存在的模型文件

# ================================================================================
#                                   训练配置
# ================================================================================
TRAIN_CONFIG = {
    "epochs": 40,
    "imgsz": 416,
    "batch": 8,
    "device": 0,
    "workers": 4,
    "amp": True,
    "verbose": True,
    "patience": 7,
    "freeze": 5,
    "save": True,
    "project": str(MODELS_SMALL),
    "exist_ok": True,
    # 数据增强
    "augment": True,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    # 优化器
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
}

# ================================================================================
#                                   辅助函数
# ================================================================================
def get_available_datasets():
    """获取small_datasets文件夹下可用的数据集"""
    if not DATASETS_SMALL.exists():
        print(f"错误: 数据集文件夹不存在: {DATASETS_SMALL}")
        return []
    
    datasets = []
    for folder in DATASETS_SMALL.iterdir():
        if folder.is_dir():
            dataset_name = folder.name
            yaml_file = folder / f"{dataset_name}.yaml"
            if yaml_file.exists():
                datasets.append(dataset_name)
    
    print(f"找到 {len(datasets)} 个有效数据集")
    return sorted(datasets)

def select_datasets(datasets):
    """选择要训练的数据集"""
    if not datasets:
        print("没有找到有效的数据集")
        return None
    
    print("\n可用数据集:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i:2d}. {dataset}")
    
    print("\n选择方式:")
    print("  输入编号 (如 '1' 或 '1,3,5')")
    print("  输入 'all' 选择所有")
    print("  输入数据集名称 (如 'basophil')")
    
    while True:
        choice = input("\n请选择: ").strip()
        
        if choice.lower() == 'all':
            print(f"选择所有 {len(datasets)} 个数据集")
            return datasets
            
        if ',' in choice:
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                selected = []
                for idx in indices:
                    if 1 <= idx <= len(datasets):
                        selected.append(datasets[idx-1])
                
                if selected:
                    print(f"选择 {len(selected)} 个数据集")
                    return selected
            except:
                print("输入格式错误，请使用 '1,3,5' 格式")
                
        elif choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(datasets):
                print(f"选择数据集: {datasets[idx-1]}")
                return [datasets[idx-1]]
                
        elif choice in datasets:
            print(f"选择数据集: {choice}")
            return [choice]
            
        else:
            print("输入无效，请重新输入")

# ================================================================================
#                                   训练函数
# ================================================================================
def train_cell_type(cell_type):
    """训练指定细胞类型模型"""
    print(f"\n开始训练 {cell_type} 模型")
    
    # 构建路径
    data_yaml = DATASETS_SMALL / cell_type / f"{cell_type}.yaml"
    output_dir = MODELS_SMALL / f"{cell_type}_train"
    
    print(f"数据集配置: {data_yaml}")
    print(f"输出目录: {output_dir}")
    
    if not data_yaml.exists():
        print(f"错误: 找不到配置文件: {data_yaml}")
        return False, 0.0
    
    try:
        # 清理旧目录
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        # 加载模型
        model = YOLO(PRETRAINED_MODEL)
        
        # 训练配置
        train_params = TRAIN_CONFIG.copy()
        train_params["data"] = str(data_yaml)
        train_params["name"] = f"{cell_type}_train"
        
        # 开始训练
        model.train(**train_params)
        
        print(f"训练完成")
        return True, get_best_map50(cell_type)
        
    except Exception as e:
        print(f"训练出错: {str(e)}")
        return False, 0.0

def get_best_map50(cell_type):
    """获取模型的最佳mAP50值"""
    results_csv = MODELS_SMALL / f"{cell_type}_train" / "results.csv"
    
    if not results_csv.exists():
        return 0.0
    
    try:
        df = pd.read_csv(results_csv)
        if 'metrics/mAP50(B)' in df.columns:
            return df['metrics/mAP50(B)'].max()
        return 0.0
    except:
        return 0.0

def evaluate_model(cell_type):
    """评估训练完成的模型"""
    best_model_path = MODELS_SMALL / f"{cell_type}_train" / "weights" / "best.pt"
    results_csv = MODELS_SMALL / f"{cell_type}_train" / "results.csv"
    
    if not best_model_path.exists():
        print(f"最佳模型文件不存在")
        return False, 0.0
    
    try:
        df = pd.read_csv(results_csv)
        
        best_map50 = df['metrics/mAP50(B)'].max() if 'metrics/mAP50(B)' in df.columns else 0.0
        final_map50 = df['metrics/mAP50(B)'].iloc[-1] if 'metrics/mAP50(B)' in df.columns else 0.0
        
        print(f"\n评估结果:")
        print(f"  最佳mAP50: {best_map50:.4f}")
        print(f"  最终mAP50: {final_map50:.4f}")
        print(f"  模型路径: {best_model_path}")
        
        return True, best_map50
    except Exception as e:
        print(f"评估出错: {str(e)}")
        return False, 0.0

# ================================================================================
#                                   主函数
# ================================================================================
def main():
    """主函数"""
    print("开始训练细胞检测模型")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查目录
    if not DATASETS_SMALL.exists():
        print(f"错误: 数据集目录不存在: {DATASETS_SMALL}")
        return
    
    # 创建模型输出目录
    MODELS_SMALL.mkdir(exist_ok=True)
    
    # 获取数据集
    datasets = get_available_datasets()
    if not datasets:
        return
    
    # 选择数据集
    selected_datasets = select_datasets(datasets)
    if not selected_datasets:
        return
    
    # 开始训练
    start_time = time.time()
    
    for i, cell_type in enumerate(selected_datasets, 1):
        print(f"\n[{i}/{len(selected_datasets)}] 训练 {cell_type}")
        train_cell_type(cell_type)
        
        # 评估
        evaluate_model(cell_type)
        
        # 如果不是最后一个，休息一下
        if i < len(selected_datasets):
            print(f"\n休息5秒...")
            time.sleep(5)
    
    # 总结
    elapsed = time.time() - start_time
    h, m, s = int(elapsed/3600), int(elapsed%3600/60), int(elapsed%60)
    
    print(f"\n训练完成! 总耗时: {h:02d}:{m:02d}:{s:02d}")
    print(f"模型保存在: {MODELS_SMALL}")

if __name__ == "__main__":
    main()