import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_video_from_folders(folder_paths, output_video="bloodcells_video.mp4", duration_per_image=2):
    """
    从多个文件夹随机选择图片合成视频
    """

    # 参数设置
    img_per_folder = 8
    fps = 0.5  # 每2秒一帧

    # 收集所有图片路径
    all_images = []
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在: {folder_path}")
            continue

        folder_name = os.path.basename(folder_path)
        # 获取文件夹中所有图片文件
        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        if len(image_files) < img_per_folder:
            print(f"警告: {folder_name} 中图片数量不足 {img_per_folder} 张，只有 {len(image_files)} 张")

        # 随机选择指定数量的图片
        selected_images = random.sample(image_files, min(img_per_folder, len(image_files)))

        for img_file in selected_images:
            all_images.append({
                'path': os.path.join(folder_path, img_file),
                'folder_name': folder_name
            })

        print(f"从 {folder_name} 选择了 {len(selected_images)} 张图片")

    if not all_images:
        print("错误: 没有找到任何图片！")
        return

    # 随机打乱图片顺序
    random.shuffle(all_images)

    # 获取第一张图片的尺寸
    first_img = cv2.imread(all_images[0]['path'])
    if first_img is None:
        print("错误: 无法读取第一张图片")
        return

    height, width = first_img.shape[:2]
    print(f"图片尺寸: {width} x {height}")

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 处理每张图片并写入视频
    for i, img_info in enumerate(all_images):
        # 读取图片
        img = cv2.imread(img_info['path'])
        if img is None:
            print(f"无法读取图片: {img_info['path']}")
            continue

        # 调整图片尺寸（如果需要）
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        # 转换为PIL图像以便添加文字
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # 添加文件夹名称文字
        try:
            # 尝试使用系统字体
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            try:
                # 尝试其他常见字体
                font = ImageFont.truetype("simhei.ttf", 40)  # 黑体
            except:
                # 使用默认字体
                font = ImageFont.load_default()

        # 在左下角添加文字
        text = img_info['folder_name']
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 计算文字位置（左下角，留一些边距）
        x = 20
        y = height - text_height - 20

        # 添加文字背景
        draw.rectangle([x - 5, y - 5, x + text_width + 5, y + text_height + 5], fill='black')
        # 添加文字
        draw.text((x, y), text, font=font, fill='white')

        # 转换回OpenCV格式
        img_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 写入多帧以控制显示时间（每张图片写入1帧，因为fps=0.5就是每2秒一帧）
        video_writer.write(img_with_text)

        print(f"进度: {i + 1}/{len(all_images)} - {img_info['folder_name']} - {os.path.basename(img_info['path'])}")

    # 释放资源
    video_writer.release()
    print(f"\n视频已生成: {output_video}")
    print(f"总共处理了 {len(all_images)} 张图片")
    print(f"视频时长: {len(all_images) * duration_per_image} 秒")


# 使用示例
if __name__ == "__main__":
    # 设置基础路径
    base_path = r"C:\Users\Eplis\Desktop\bloodcells_dataset"

    # 8个文件夹的完整路径
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

    # 创建视频
    create_video_from_folders(
        folder_paths=folders,
        output_video="bloodcells_random_video.mp4",
        duration_per_image=2  # 每张图片显示2秒
    )