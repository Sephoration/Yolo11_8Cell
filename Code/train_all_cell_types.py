# 批量训练所有细胞类型的YOLO11n模型，并评估性能
from ultralytics import YOLO
import os
import time
import pandas as pd
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import (
    PROJECT_ROOT,
    DATASETS_SMALL,
    MODELS_SMALL,
    YOLO11N_MODEL,
    YOLO11M_MODEL
)


def get_available_datasets():
    """获取small_datasets文件夹下可用的数据集"""
    print(f"\n搜索数据集文件夹: {DATASETS_SMALL}")
    
    if not os.path.exists(DATASETS_SMALL):
        print(f"❌ 错误: 数据集文件夹不存在: {DATASETS_SMALL}")
        return []
    
    datasets = []
    # 列出small_datasets下的所有子文件夹
    for folder in glob.glob(f"{DATASETS_SMALL}/*"):
        if os.path.isdir(folder):
            dataset_name = os.path.basename(folder)
            # 检查是否有对应的YAML文件
            yaml_file = os.path.join(folder, f"{dataset_name}.yaml")
            if os.path.exists(yaml_file):
                datasets.append(dataset_name)
            else:
                print(f"⚠️  警告: {dataset_name} 缺少YAML配置文件")
    
    print(f"找到 {len(datasets)} 个有效数据集")
    return sorted(datasets)


def select_datasets_interactive(datasets):
    """交互式选择数据集"""
    if not datasets:
        print("❌ 没有找到有效的数据集，请检查small_datasets文件夹")
        return None
    
    print("\n" + "="*60)
    print("small_datasets文件夹下的可用数据集：")
    print("="*60)
    
    for i, dataset in enumerate(datasets, 1):
        print(f"{i:2d}. {dataset}")
    
    print("\n选择方式：")
    print("  • 输入编号（如 '1' 或 '1,3,5'）")
    print("  • 输入 'all' 选择所有数据集")
    print("  • 输入数据集名称（如 'basophil'）")
    print("  • 输入 'exit' 退出程序")
    
    while True:
        choice = input("\n请选择要训练的数据集: ").strip()
        
        if choice.lower() == 'exit':
            print("👋 退出程序")
            return None
            
        if choice.lower() == 'all':
            print(f"✅ 选择所有 {len(datasets)} 个数据集")
            return datasets
            
        if ',' in choice:
            # 多选：'1,3,5'
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                selected = []
                for idx in indices:
                    if 1 <= idx <= len(datasets):
                        selected.append(datasets[idx-1])
                    else:
                        print(f"❌ 编号 {idx} 超出范围 (1-{len(datasets)})")
                
                if selected:
                    print(f"✅ 选择 {len(selected)} 个数据集: {', '.join(selected)}")
                    return selected
                else:
                    print("❌ 没有选择任何有效的数据集")
            except ValueError:
                print("❌ 输入格式错误，请使用 '1,3,5' 格式")
                
        elif choice.isdigit():
            # 单选：'3'
            idx = int(choice)
            if 1 <= idx <= len(datasets):
                print(f"✅ 选择数据集: {datasets[idx-1]}")
                return [datasets[idx-1]]
            else:
                print(f"❌ 编号 {idx} 超出范围 (1-{len(datasets)})")
                
        elif choice in datasets:
            # 直接输入名称
            print(f"✅ 选择数据集: {choice}")
            return [choice]
            
        else:
            print("❌ 输入无效，请重新输入")


def select_model_interactive():
    """交互式选择预训练模型"""
    print("\n" + "="*60)
    print("选择预训练模型：")
    print("="*60)
    print("1. yolo11n.pt - 轻量版 (较小，训练速度快)")
    print("2. yolo11m.pt - 中量版 (中等大小，精度较高)")
    
    while True:
        choice = input("\n请选择模型 (输入 1 或 2): ").strip()
        
        if choice == '1':
            if os.path.exists(YOLO11N_MODEL):
                print(f"✅ 选择模型: yolo11n.pt (本地文件)")
                return YOLO11N_MODEL
            else:
                print("⚠️  本地yolo11n.pt不存在，将使用在线版本")
                return "yolo11n.pt"
                
        elif choice == '2':
            if os.path.exists(YOLO11M_MODEL):
                print(f"✅ 选择模型: yolo11m.pt (本地文件)")
                return YOLO11M_MODEL
            else:
                print("⚠️  本地yolo11m.pt不存在，将使用在线版本")
                return "yolo11m.pt"
                
        elif choice.lower() == 'exit':
            print("👋 退出程序")
            return None
            
        else:
            print("❌ 请输入 1 或 2 (或输入 'exit' 退出)")


def train_cell_type(cell_type, model_path):
    """训练指定细胞类型模型"""
    print(f"\n{'='*60}")
    print(f"开始训练 {cell_type} 模型")
    print(f"{'='*60}")

    # 直接使用配置文件中的常量构建路径
    data_yaml = f"{DATASETS_SMALL}/{cell_type}/{cell_type}.yaml"
    print(f"📁 数据集配置文件: {data_yaml}")
    
    # 构建模型保存路径
    output_dir = f"{MODELS_SMALL}/{cell_type}_train"
    print(f"💾 模型输出目录: {output_dir}")
    
    # 打印所有关键路径信息
    print(f"🤖 模型文件: {model_path}")
    
    # 检查YAML文件
    if not os.path.exists(data_yaml):
        print(f"❌ 错误: 找不到 {cell_type} 的配置文件: {data_yaml}")
        return False, 0.0

    # 检查模型文件是否存在（如果不是在线下载）
    if not model_path.startswith("yolo11") and not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        return False, 0.0

    # 加载模型并训练
    try:
        print(f"⏳ 加载模型: {model_path}")
        model = YOLO(model_path)

        # 清理可能存在的旧训练文件夹
        train_dir = os.path.join(output_dir, f"{cell_type}_train")
        if os.path.exists(train_dir):
            print(f"🧹 清理旧的训练文件夹: {train_dir}")
            import shutil
            shutil.rmtree(train_dir)

        # 开始训练
        print(f"🚀 开始训练 {cell_type}，使用 {os.path.basename(model_path)} 模型")
        model.train(
            data=data_yaml,
            epochs=30,
            imgsz=416,
            batch=6,
            device=0,
            workers=0,
            save=True,
            project=output_dir,
            name=f'{cell_type}_train',
            amp=False,
            verbose=True,
            patience=10,
            freeze=5
        )

        print(f"✅ {cell_type} 训练完成")
        return True, get_best_map50(cell_type)

    except Exception as e:
        print(f"❌ {cell_type} 训练出错: {str(e)}")
        return False, 0.0


def get_best_map50(cell_type):
    """获取模型的最佳mAP50值"""
    # 直接使用配置文件中的常量构建路径
    results_csv = f"{MODELS_SMALL}/{cell_type}_train/results.csv"
    
    if not os.path.exists(results_csv):
        print(f"⚠️  警告: 结果文件不存在: {results_csv}")
        return 0.0
    
    try:
        df = pd.read_csv(results_csv)
        if 'metrics/mAP50(B)' in df.columns:
            return df['metrics/mAP50(B)'].max()
        return 0.0
    except Exception as e:
        print(f"读取 {cell_type} 结果文件出错: {str(e)}")
        return 0.0


def evaluate_model(cell_type):
    """评估训练完成的模型"""
    # 直接使用配置文件中的常量构建路径
    best_model_path = f"{MODELS_SMALL}/{cell_type}_train/weights/best.pt"
    results_csv = f"{MODELS_SMALL}/{cell_type}_train/results.csv"
    
    print(f"🔍 评估模型路径: {best_model_path}")
    
    if not os.path.exists(best_model_path):
        print(f"⚠️  {cell_type} 最佳模型文件不存在")
        return False, 0.0, None
    
    try:
        df = pd.read_csv(results_csv)
        epochs_completed = len(df)
        final_map50 = df['metrics/mAP50(B)'].iloc[-1] if 'metrics/mAP50(B)' in df.columns else 0.0
        best_map50 = df['metrics/mAP50(B)'].max() if 'metrics/mAP50(B)' in df.columns else 0.0
        final_precision = df['metrics/precision(B)'].iloc[-1] if 'metrics/precision(B)' in df.columns else 0.0
        final_recall = df['metrics/recall(B)'].iloc[-1] if 'metrics/recall(B)' in df.columns else 0.0
        
        print(f"\n📊 {cell_type} 模型评估:")
        print(f"  🎯 训练轮数: {epochs_completed}/30")
        print(f"  🥇 最佳mAP50: {best_map50:.4f}")
        print(f"  🏁 最终mAP50: {final_map50:.4f}")
        print(f"  📏 最终精确率: {final_precision:.4f}")
        print(f"  🔍 最终召回率: {final_recall:.4f}")
        print(f"  💾 模型路径: {best_model_path}")
        
        return True, best_map50, best_model_path
    except Exception as e:
        print(f"评估 {cell_type} 模型出错: {str(e)}")
        return False, 0.0, None


def train_selected_datasets(selected_datasets, model_path):
    """训练选中的数据集"""
    start_time = time.time()
    training_results = {}
    
    print(f"\n{'='*60}")
    print(f"开始训练 {len(selected_datasets)} 个数据集")
    print(f"使用模型: {os.path.basename(model_path)}")
    print(f"{'='*60}")
    
    # 训练所有选中的细胞类型
    for i, cell_type in enumerate(selected_datasets, 1):
        print(f"\n[{i}/{len(selected_datasets)}] 训练 {cell_type}")
        success, best_map50 = train_cell_type(cell_type, model_path)
        training_results[cell_type] = {"success": success, "mAP50": best_map50}
        
        # 如果不是最后一个，休息一下
        if i < len(selected_datasets):
            print(f"\n⏸️  休息10秒，准备下一个训练任务...")
            time.sleep(10)
    
    # 评估所有训练完成的模型
    print(f"\n{'='*60}")
    print(f"开始评估所有训练完成的模型")
    print(f"{'='*60}")
    
    evaluation_report = []
    for cell_type in selected_datasets:
        if training_results[cell_type]["success"]:
            success, best_map50, model_path_result = evaluate_model(cell_type)
            if success:
                evaluation_report.append((cell_type, best_map50, model_path_result))
        else:
            print(f"⚠️  {cell_type} 训练失败，跳过评估")
    
    # 生成最终报告
    total_time = time.time() - start_time
    h, m, s = int(total_time / 3600), int((total_time % 3600) / 60), int(total_time % 60)
    
    print(f"\n{'='*60}")
    print(f"训练和评估完成!")
    print(f"{'='*60}")
    print(f"⏱️  总耗时: {h}小时 {m}分钟 {s}秒")
    print(f"\n📈 性能评估汇总:")
    print("-" * 60)
    print(f"{'细胞类型':<15} {'最佳mAP50':<12} {'状态'}")
    print("-" * 60)
    
    for cell_type, map50, _ in sorted(evaluation_report, key=lambda x: x[1], reverse=True):
        status = "✅ 优秀" if map50 >= 0.90 else "✅ 良好" if map50 >= 0.70 else "⚠️  一般" if map50 >= 0.50 else "❌ 较差"
        print(f"{cell_type:<15} {map50:.4f}        {status}")
    
    # 计算平均性能
    if evaluation_report:
        avg_map50 = sum([x[1] for x in evaluation_report]) / len(evaluation_report)
        print("-" * 60)
        print(f"{'平均mAP50':<15} {avg_map50:.4f}")
    
    print(f"\n💡 注意: 训练结果保存在 {MODELS_SMALL}/[cell_type]_train 目录中")


def main():
    """主函数：交互式选择数据集和模型进行训练"""
    print(f"\n{'='*60}")
    print(f"YOLO细胞检测模型训练系统")
    print(f"{'='*60}")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据集目录: {DATASETS_SMALL}")
    
    # 1. 获取可用的数据集
    datasets = get_available_datasets()
    if not datasets:
        print("❌ 无法继续，请检查数据集配置")
        return
    
    # 2. 让用户选择数据集
    selected_datasets = select_datasets_interactive(datasets)
    if selected_datasets is None:
        return
    
    # 3. 让用户选择模型
    model_path = select_model_interactive()
    if model_path is None:
        return
    
    # 4. 开始训练选中的数据集
    train_selected_datasets(selected_datasets, model_path)
    
    print(f"\n🎉 所有训练任务完成!")
    print(f"💾 模型保存位置: {MODELS_SMALL}/")
    print(f"📊 详细结果查看各个训练文件夹内的results.csv文件")


if __name__ == "__main__":
    main()