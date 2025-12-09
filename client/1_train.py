from ultralytics import YOLO
import torch
import os


class EarlyStopping:
    """早停法类"""

    def __init__(self, patience=4, min_delta=0.001, verbose=True):
        self.patience = patience
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.early_stop = False

    def __call__(self, val_loss, trainer):
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"Validation loss improved from {self.best_loss:.4f} to {val_loss:.4f}")
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.verbose:
                print(f"Validation loss did not improve. Patience counter: {self.patience_counter}/{self.patience}")

            if self.patience_counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")
                trainer.stop()


def early_stopping_callback(trainer):
    """早停法回调函数"""
    # 确保早停实例存在
    if not hasattr(trainer, 'early_stopping'):
        trainer.early_stopping = EarlyStopping(patience=4, min_delta=0.001)

    # 获取验证损失
    val_loss = trainer.metrics.get('val/loss', None)
    if val_loss is not None:
        trainer.early_stopping(val_loss, trainer)


def setup_environment():
    """设置训练环境"""
    # 设置环境变量禁用证书验证
    os.environ['CURL_CA_BUNDLE'] = ''

    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")


def check_existing_training():
    """检查是否存在之前的训练"""
    weights_path = r'D:\Code\YOLO_8Cell\runs\classify_1\weights\last.pt'
    if os.path.exists(weights_path):
        print("检测到之前的训练，将恢复训练...")
        return weights_path
    else:
        print("未检测到之前的训练，将从零开始训练...")
        return None


def main():
    """主训练函数"""
    # 设置环境
    setup_environment()

    try:
        # 检查是否存在之前的训练
        existing_model_path = check_existing_training()

        if existing_model_path:
            # 恢复训练
            model = YOLO(existing_model_path)
            resume = True
            print(f"从第{model.ckpt['epoch'] + 1}个epoch恢复训练")
        else:
            # 从零开始训练
            model = YOLO('yolov8n-cls.yaml')
            resume = False
            print("开始新的训练")

        # 添加早停法回调函数
        model.add_callback('on_epoch_end', early_stopping_callback)

        # 训练模型
        results = model.train(
            data=r'D:\Code\YOLO_8Cell\datasets8',
            epochs=50,
            imgsz=224,
            batch=32,  # 减小batch size防止内存溢出
            project=r'D:\Code\YOLO_8Cell\runs',
            name='classify_1',
            resume=resume,  # 自动决定是否恢复训练
            patience=0,  # 禁用YOLO内置的早停，使用自定义的
            save=True,  # 保存最佳模型
            exist_ok=True,  # 允许覆盖现有运行
            workers=4,  # 减少workers防止内存问题
            lr0=0.0001 if resume else 0.01,  # 恢复训练时使用更小的学习率
        )

        print("训练完成!")
        return results

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return None


if __name__ == '__main__':
    main()