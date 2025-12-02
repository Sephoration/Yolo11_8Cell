from ultralytics import YOLO
import os
import time
import pandas as pd


def train_cell_type(cell_type):
    """训练指定细胞类型模型"""
    print(f"\n{'=' * 60}")
    print(f"开始训练 {cell_type} 模型")
    print(f"{'=' * 60}")

    # 获取当前脚本所在目录（small_models）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取YOLO_8Cell根目录（small_models的父目录）
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))
    
    print(f"当前脚本目录: {script_dir}")
    print(f"项目根目录: {base_dir}")
    
    # 构建模型和数据集路径
    model_path = os.path.join(script_dir, "yolo11n.pt")
    data_yaml = os.path.join(base_dir, "small_datasets", cell_type, f"{cell_type}.yaml")
    output_dir = script_dir
    
    # 打印所有关键路径信息
    print(f"模型文件路径: {model_path}")
    print(f"数据集配置路径: {data_yaml}")
    print(f"输出目录: {output_dir}")
    
    # 检查YAML文件
    if not os.path.exists(data_yaml):
        alt_yaml = os.path.join(base_dir, "small_datasets", cell_type, "data.yaml")
        if os.path.exists(alt_yaml):
            data_yaml = alt_yaml
            print(f"使用替代配置文件: {alt_yaml}")
        else:
            print(f"错误: 找不到 {cell_type} 的配置文件")
            return False, 0.0

    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return False, 0.0

    # 加载模型并训练
    try:
        print(f"加载模型: {model_path}")
        print(f"使用数据集配置: {data_yaml}")
        model = YOLO(model_path)

        # 清理可能存在的旧训练文件夹
        train_dir = os.path.join(output_dir, f"{cell_type}_train")
        if os.path.exists(train_dir):
            print(f"清理旧的训练文件夹: {train_dir}")
            import shutil
            shutil.rmtree(train_dir)

        # 开始训练
        print(f"开始训练 {cell_type}，使用yolo11n模型")
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
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建结果文件路径
    results_csv = os.path.join(script_dir, f"{cell_type}_train", "results.csv")
    
    if not os.path.exists(results_csv):
        print(f"警告: 结果文件不存在: {results_csv}")
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
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建评估所需的路径
    train_dir = os.path.join(script_dir, f"{cell_type}_train")
    best_model_path = os.path.join(train_dir, "weights", "best.pt")
    results_csv = os.path.join(train_dir, "results.csv")
    
    print(f"评估模型路径: {best_model_path}")
    print(f"结果文件路径: {results_csv}")
    
    if not os.path.exists(best_model_path):
        print(f"⚠️ {cell_type} 最佳模型文件不存在")
        return False, 0.0, None
    
    try:
        df = pd.read_csv(results_csv)
        epochs_completed = len(df)
        final_map50 = df['metrics/mAP50(B)'].iloc[-1] if 'metrics/mAP50(B)' in df.columns else 0.0
        best_map50 = df['metrics/mAP50(B)'].max() if 'metrics/mAP50(B)' in df.columns else 0.0
        final_precision = df['metrics/precision(B)'].iloc[-1] if 'metrics/precision(B)' in df.columns else 0.0
        final_recall = df['metrics/recall(B)'].iloc[-1] if 'metrics/recall(B)' in df.columns else 0.0
        
        print(f"\n{cell_type} 模型评估:")
        print(f"  训练轮数: {epochs_completed}/30")
        print(f"  最佳mAP50: {best_map50:.4f}")
        print(f"  最终mAP50: {final_map50:.4f}")
        print(f"  最终精确率: {final_precision:.4f}")
        print(f"  最终召回率: {final_recall:.4f}")
        print(f"  模型路径: {best_model_path}")
        
        return True, best_map50, best_model_path
    except Exception as e:
        print(f"评估 {cell_type} 模型出错: {str(e)}")
        return False, 0.0, None


def main():
    """主函数：批量训练并评估所有细胞类型"""
    cell_types = [
        "basophil", "eosinophil", "erythroblast", "ig",
        "lymphocyte", "monocyte", "neutrophil", "platelet"
    ]

    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取YOLO_8Cell根目录
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))
    
    # 构建项目相关目录路径
    small_datasets_dir = os.path.join(base_dir, "small_datasets")
    full_datasets_dir = os.path.join(base_dir, "full_datasets")

    print(f"\n{'=' * 60}")
    print(f"开始批量训练和评估 {len(cell_types)} 种细胞类型")
    print(f"使用模型: yolo11n")
    print(f"当前脚本目录: {script_dir}")
    print(f"项目根目录: {base_dir}")
    print(f"small_datasets目录: {small_datasets_dir}")
    print(f"full_datasets目录: {full_datasets_dir}")
    print(f"{'=' * 60}")

    start_time = time.time()
    training_results = {}

    # 训练所有细胞类型
    for i, cell_type in enumerate(cell_types, 1):
        print(f"\n[{i}/{len(cell_types)}] 训练 {cell_type}")
        success, best_map50 = train_cell_type(cell_type)
        training_results[cell_type] = {"success": success, "mAP50": best_map50}
        
        if i < len(cell_types):
            print("\n休息20秒，准备下一个训练任务...")
            time.sleep(20)

    # 评估所有模型并生成报告
    print(f"\n{'=' * 60}")
    print(f"开始评估所有训练完成的模型")
    print(f"{'=' * 60}")
    
    evaluation_report = []
    for cell_type in cell_types:
        if training_results[cell_type]["success"]:
            success, best_map50, model_path = evaluate_model(cell_type)
            if success:
                evaluation_report.append((cell_type, best_map50, model_path))
        else:
            print(f"⚠️ {cell_type} 训练失败，跳过评估")

    # 生成最终报告
    total_time = time.time() - start_time
    h, m, s = int(total_time / 3600), int((total_time % 3600) / 60), int(total_time % 60)

    print(f"\n{'=' * 60}")
    print(f"批量训练和评估完成!")
    print(f"{'=' * 60}")
    print(f"总耗时: {h}小时 {m}分钟 {s}秒")
    print(f"\n性能评估汇总:")
    print("-" * 60)
    print(f"{'细胞类型':<15} {'最佳mAP50':<12} {'状态'}")
    print("-" * 60)
    
    for cell_type, map50, _ in sorted(evaluation_report, key=lambda x: x[1], reverse=True):
        status = "✅ 优秀" if map50 >= 0.90 else "✅ 良好" if map50 >= 0.70 else "⚠️ 一般" if map50 >= 0.50 else "❌ 较差"
        print(f"{cell_type:<15} {map50:.4f}        {status}")

    # 计算平均性能
    if evaluation_report:
        avg_map50 = sum([x[1] for x in evaluation_report]) / len(evaluation_report)
        print("-" * 60)
        print(f"{'平均mAP50':<15} {avg_map50:.4f}")

    print(f"\n注意: 训练结果保存在 small_models/{cell_type}_train 目录中")


if __name__ == "__main__":
    main()