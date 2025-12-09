import os
import random
import cv2
import numpy as np


def create_video_from_folders(folder_paths,
                              output_video="bloodcells_video.mp4",
                              duration_per_image=2):
    """
    从多个文件夹随机选择图片合成视频，并对图片进行处理使其不同于原图
    """
    # 参数设置
    img_per_folder = 8
    fps = 1 / duration_per_image  # 例如 2 秒一帧 -> 0.5 fps

    # 收集所有图片路径
    all_images = []
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在: {folder_path}")
            continue

        folder_name = os.path.basename(folder_path)
        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        if len(image_files) < img_per_folder:
            print(f"警告: {folder_name} 中图片数量不足 {img_per_folder} 张，只有 {len(image_files)} 张")

        selected_images = random.sample(image_files,
                                        min(img_per_folder, len(image_files)))
        all_images.extend([(os.path.join(folder_path, f), folder_name) for f in selected_images])
        print(f"从 {folder_name} 选择了 {len(selected_images)} 张图片")

    if not all_images:
        print("错误: 没有找到任何图片！")
        return

    random.shuffle(all_images)

    # 读取第一张图，获取尺寸
    first_img_path = all_images[0][0]
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print("错误: 无法读取第一张图片")
        return
    height, width = first_img.shape[:2]
    print(f"原图尺寸: {width} x {height}")

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 图像处理函数
    def process_image(img, folder_name):
        """对图像进行处理使其不同于原图"""
        processed_img = img.copy()

        # 随机选择一种或多种处理方式
        processing_methods = [
            adjust_brightness_contrast,
            add_gaussian_noise,
            apply_gaussian_blur,
            rotate_image,
            flip_image
        ]

        # 随机应用1-3种处理
        num_processing = random.randint(1, 3)
        selected_methods = random.sample(processing_methods, num_processing)

        for method in selected_methods:
            processed_img = method(processed_img)

        # 添加类别名称标签
        processed_img = add_text_label(processed_img, folder_name)

        return processed_img

    def adjust_brightness_contrast(img):
        """调整亮度和对比度"""
        brightness = random.randint(-30, 30)
        contrast = random.uniform(0.7, 1.3)

        img = img.astype(np.float32)
        img = img * contrast + brightness
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def add_gaussian_noise(img):
        """添加高斯噪声"""
        mean = 0
        sigma = random.randint(5, 15)
        gaussian = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
        noisy_img = cv2.add(img, gaussian)
        return noisy_img

    def apply_gaussian_blur(img):
        """应用高斯模糊"""
        kernel_size = random.choice([3, 5])
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def rotate_image(img):
        """随机旋转图像"""
        angle = random.randint(-10, 10)
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
        return rotated_img

    def flip_image(img):
        """随机翻转图像"""
        flip_code = random.choice([-1, 0, 1])  # -1: 水平和垂直, 0: 垂直, 1: 水平
        if flip_code != 2:  # 2表示不翻转
            return cv2.flip(img, flip_code)
        return img

    def add_text_label(img, text):
        """在图像上添加类别名称"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (255, 255, 255)  # 白色文字

        # 获取文字尺寸
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # 计算文字位置（左上角）
        text_x = 10
        text_y = text_size[1] + 10

        # 添加文字背景
        bg_color = (0, 0, 0)  # 黑色背景
        cv2.rectangle(img,
                      (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + 5),
                      bg_color, -1)

        # 添加文字
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

        return img

    # 逐张处理并写入
    for idx, (img_path, folder_name) in enumerate(all_images, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        # 调整尺寸
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        # 处理图像
        processed_img = process_image(img, folder_name)

        # 写入视频
        video_writer.write(processed_img)
        print(f"进度: {idx}/{len(all_images)} - {folder_name}/{os.path.basename(img_path)}")

    video_writer.release()
    print(f"\n视频已生成: {output_video}")
    print(f"总共处理了 {len(all_images)} 张图片")
    print(f"视频时长: {len(all_images) * duration_per_image} 秒")


# 使用示例
if __name__ == "__main__":
    base_path = r"D:\Code\YOLO_8Cell\datasets"
    folders = [
        os.path.join(base_path, "basophil"),
        os.path.join(base_path, "eosinophil"),
        os.path.join(base_path, "erythroblast"),
        os.path.join(base_path, "ig"),
        os.path.join(base_path, "lymphocyte"),
        os.path.join(base_path, "monocyte"),
        os.path.join(base_path, "neutrophil"),
        os.path.join(base_path, "platelet")
    ]

    create_video_from_folders(
        folder_paths=folders,
        output_video="processed_bloodcells_video.mp4",
        duration_per_image=2
    )