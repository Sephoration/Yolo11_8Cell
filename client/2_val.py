from ultralytics import YOLO


def main():
    model = YOLO(r'D:\Code\YOLO_8Cell\runs\classify_1\weights\best.pt')

    # 验证并获取指标
    metrics = model.val(data=r'D:\Code\YOLO_8Cell\datasets8')

    # 正确的指标输出
    print('=' * 50)
    print('模型验证结果:')
    print('=' * 50)
    print(f'Top-1 准确率: {metrics.top1:.4f} ({metrics.top1 * 100:.2f}%)')
    print(f'Top-5 准确率: {metrics.top5:.4f} ({metrics.top5 * 100:.2f}%)')
    print(f'验证图片数量: 2400')

    # 正确读取推理速度（从字典中获取）
    inference_speed = metrics.speed['inference']  # 关键修正！
    print(f'推理速度: {inference_speed:.2f} ms/张')
    print('=' * 50)


if __name__ == '__main__':
    main()


