import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from ultralytics import YOLO
import os
import cv2
import threading
from PIL import Image, ImageTk
import time
import queue


class SimpleCellClassifier:
    def __init__(self):
        # ====== 初始化变量 ======
        self.current_mode = None  # 当前模式：'image', 'video', 'camera'
        self.video_processing = False  # 视频处理状态
        self.camera_processing = False  # 摄像头处理状态
        self.cap = None  # 摄像头对象
        self.current_frame = None  # 当前帧
        self.model = None  # 模型对象
        self.video_cap = None  # 视频捕获对象

        # ====== 创建UI更新队列 ======
        self.ui_queue = queue.Queue()

        # ====== 细胞类别中英文对照 ======
        self.cell_classes = {
            'basophil': '嗜碱性粒细胞',
            'eosinophil': '嗜酸性粒细胞',
            'erythroblast': '成红细胞',
            'ig': '未成熟粒细胞',
            'lymphocyte': '淋巴细胞',
            'monocyte': '单核细胞',
            'neutrophil': '中性粒细胞',
            'platelet': '血小板'
        }

        self.setup_gui()
        self.start_ui_update_processor()

    def setup_gui(self):
        """创建GUI界面"""
        # ====== 主窗口设置 ======
        self.root = tk.Tk()
        self.root.title("细胞分类系统 v1.0")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # 设置窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ====== 标题区域 ======
        title_label = tk.Label(self.root, text="🦠 细胞图像分类系统",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=20)

        # ====== 模式选择按钮区域 ======
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=15)

        btn_image = tk.Button(btn_frame, text="📷 图片识别",
                              font=("Arial", 12), width=15, height=2,
                              command=self.predict_image)
        btn_image.pack(side=tk.LEFT, padx=10)

        btn_video = tk.Button(btn_frame, text="🎥 视频识别",
                              font=("Arial", 12), width=15, height=2,
                              command=self.predict_video)
        btn_video.pack(side=tk.LEFT, padx=10)

        btn_camera = tk.Button(btn_frame, text="📹 摄像头识别",
                               font=("Arial", 12), width=15, height=2,
                               command=self.predict_camera)
        btn_camera.pack(side=tk.LEFT, padx=10)

        # ====== 文件路径显示区域 ======
        self.file_label = tk.Label(self.root, text="请选择识别模式",
                                   font=("Arial", 10), fg="blue", wraplength=750)
        self.file_label.pack(pady=10)

        # ====== 主内容区域 - 初始为空 ======
        self.main_content_frame = tk.Frame(self.root)
        self.main_content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # ====== 状态栏 ======
        self.status_label = tk.Label(self.root, text="准备就绪",
                                     font=("Arial", 9), fg="gray", bd=1, relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # 初始显示欢迎信息
        self.show_welcome_message()

    def start_ui_update_processor(self):
        """启动UI更新处理器"""

        def process_ui_updates():
            while True:
                try:
                    # 从队列获取更新任务
                    task, args, kwargs = self.ui_queue.get(timeout=0.1)
                    if task == "stop":
                        break
                    try:
                        if hasattr(self, task):
                            getattr(self, task)(*args, **kwargs)
                    except Exception as e:
                        print(f"UI任务执行错误: {e}")
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"UI更新处理错误: {e}")

        self.ui_thread = threading.Thread(target=process_ui_updates, daemon=True)
        self.ui_thread.start()

    def safe_ui_update(self, task, *args, **kwargs):
        """线程安全的UI更新方法"""
        try:
            self.ui_queue.put((task, args, kwargs))
        except Exception as e:
            print(f"添加UI任务失败: {e}")

    def on_closing(self):
        """窗口关闭事件处理"""
        # 停止所有处理
        self.video_processing = False
        self.camera_processing = False

        # 释放摄像头和视频
        if self.cap:
            self.cap.release()
        if self.video_cap:
            self.video_cap.release()

        # 停止UI更新处理器
        try:
            self.ui_queue.put(("stop", [], {}))
        except:
            pass

        # 关闭窗口
        self.root.quit()
        self.root.destroy()

    def clear_main_content(self):
        """清空主内容区域"""
        for widget in self.main_content_frame.winfo_children():
            widget.destroy()

    def show_welcome_message(self):
        """显示欢迎信息"""
        self.clear_main_content()
        welcome_text = """欢迎使用细胞分类系统！

请点击上方按钮选择识别模式：

📷 图片识别 - 识别单张细胞图片，显示详细分析结果
🎥 视频识别 - 识别视频中的细胞，实时显示识别结果  
📹 摄像头识别 - 实时摄像头识别细胞

选择相应模式后，界面将显示对应的功能区域。"""

        welcome_label = tk.Label(self.main_content_frame, text=welcome_text,
                                 font=("Arial", 11), justify=tk.LEFT, fg="darkblue")
        welcome_label.pack(expand=True, pady=50)

    def setup_image_mode(self):
        """设置图片识别模式界面"""
        self.clear_main_content()

        # ====== 图片结果显示区域 ======
        image_frame = tk.Frame(self.main_content_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)

        result_frame = tk.Frame(image_frame)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(result_frame, text="识别结果:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.result_text = scrolledtext.ScrolledText(
            result_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            font=("Consolas", 10)
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def setup_video_mode(self):
        """设置视频识别模式界面"""
        self.clear_main_content()

        # ====== 视频和实时结果主容器 ======
        video_main_frame = tk.Frame(self.main_content_frame)
        video_main_frame.pack(fill=tk.BOTH, expand=True)

        # ====== 左侧视频区域 ======
        left_video_frame = tk.Frame(video_main_frame)
        left_video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # 视频显示区域
        video_display_frame = tk.Frame(left_video_frame, bg="black", relief=tk.RAISED, bd=2)
        video_display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.video_label = tk.Label(video_display_frame, bg="black", text="视频预览区域\n\n选择视频文件后点击播放",
                                    fg="white", font=("Arial", 12), justify=tk.CENTER)
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # ====== 视频控制按钮区域 ======
        self.video_control_frame = tk.Frame(left_video_frame)
        self.video_control_frame.pack(pady=5)

        self.btn_play = tk.Button(self.video_control_frame, text="▶ 播放",
                                  font=("Arial", 10), width=8, height=1,
                                  command=self.play_video, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=3)

        self.btn_pause = tk.Button(self.video_control_frame, text="⏸ 暂停",
                                   font=("Arial", 10), width=8, height=1,
                                   command=self.pause_video, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=3)

        self.btn_stop = tk.Button(self.video_control_frame, text="⏹ 停止",
                                  font=("Arial", 10), width=8, height=1,
                                  command=self.stop_video, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=3)

        # ====== 右侧实时结果显示区域 ======
        right_result_frame = tk.Frame(video_main_frame, width=300, relief=tk.RAISED, bd=2)
        right_result_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_result_frame.pack_propagate(False)

        # 实时结果标题
        result_title = tk.Label(right_result_frame, text="🎯 实时识别结果",
                                font=("Arial", 14, "bold"), bg="lightgray")
        result_title.pack(fill=tk.X, pady=10)

        # 主要结果显示区域
        result_content = tk.Frame(right_result_frame)
        result_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # 当前识别结果
        tk.Label(result_content, text="当前识别:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.main_class_label = tk.Label(result_content, text="无",
                                         font=("Arial", 16, "bold"), fg="red",
                                         wraplength=250, height=2)
        self.main_class_label.pack(fill=tk.X, pady=5)

        self.confidence_label = tk.Label(result_content, text="置信度: --",
                                         font=("Arial", 11))
        self.confidence_label.pack(pady=5)

        # 分隔线
        separator = tk.Frame(result_content, height=2, bg="gray")
        separator.pack(fill=tk.X, pady=15)

        # 详细信息
        tk.Label(result_content, text="详细信息:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.detail_text = tk.Text(result_content, wrap=tk.WORD, width=30, height=10,
                                   font=("Consolas", 9), state=tk.DISABLED)
        self.detail_text.pack(fill=tk.BOTH, expand=True)

        # 进度信息
        self.progress_label = tk.Label(result_content, text="状态: 等待开始",
                                       font=("Arial", 9), fg="darkgreen")
        self.progress_label.pack(pady=10)

    def setup_camera_mode(self):
        """设置摄像头识别模式界面"""
        self.clear_main_content()

        # ====== 摄像头和实时结果主容器 ======
        camera_main_frame = tk.Frame(self.main_content_frame)
        camera_main_frame.pack(fill=tk.BOTH, expand=True)

        # ====== 左侧摄像头区域 ======
        left_camera_frame = tk.Frame(camera_main_frame)
        left_camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # 摄像头显示区域
        camera_display_frame = tk.Frame(left_camera_frame, bg="black", relief=tk.RAISED, bd=2)
        camera_display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.camera_label = tk.Label(camera_display_frame, bg="black", text="摄像头预览区域\n\n点击开始进行实时识别",
                                     fg="white", font=("Arial", 12), justify=tk.CENTER)
        self.camera_label.pack(expand=True, fill=tk.BOTH)

        # ====== 摄像头控制按钮区域 ======
        self.camera_control_frame = tk.Frame(left_camera_frame)
        self.camera_control_frame.pack(pady=5)

        self.btn_camera_start = tk.Button(self.camera_control_frame, text="▶ 开始",
                                          font=("Arial", 10), width=8, height=1,
                                          command=self.start_camera, state=tk.NORMAL)
        self.btn_camera_start.pack(side=tk.LEFT, padx=3)

        self.btn_camera_stop = tk.Button(self.camera_control_frame, text="⏹ 停止",
                                         font=("Arial", 10), width=8, height=1,
                                         command=self.stop_camera, state=tk.DISABLED)
        self.btn_camera_stop.pack(side=tk.LEFT, padx=3)

        # ====== 右侧实时结果显示区域 ======
        right_result_frame = tk.Frame(camera_main_frame, width=300, relief=tk.RAISED, bd=2)
        right_result_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_result_frame.pack_propagate(False)

        # 实时结果标题
        result_title = tk.Label(right_result_frame, text="🎯 实时识别结果",
                                font=("Arial", 14, "bold"), bg="lightgray")
        result_title.pack(fill=tk.X, pady=10)

        # 主要结果显示区域
        result_content = tk.Frame(right_result_frame)
        result_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # 当前识别结果
        tk.Label(result_content, text="当前识别:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.camera_main_class_label = tk.Label(result_content, text="无",
                                                font=("Arial", 16, "bold"), fg="red",
                                                wraplength=250, height=2)
        self.camera_main_class_label.pack(fill=tk.X, pady=5)

        self.camera_confidence_label = tk.Label(result_content, text="置信度: --",
                                                font=("Arial", 11))
        self.camera_confidence_label.pack(pady=5)

        # 分隔线
        separator = tk.Frame(result_content, height=2, bg="gray")
        separator.pack(fill=tk.X, pady=15)

        # 详细信息
        tk.Label(result_content, text="详细信息:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.camera_detail_text = tk.Text(result_content, wrap=tk.WORD, width=30, height=10,
                                          font=("Consolas", 9), state=tk.DISABLED)
        self.camera_detail_text.pack(fill=tk.BOTH, expand=True)

        # 状态信息
        self.camera_status_label = tk.Label(result_content, text="状态: 等待开始",
                                            font=("Arial", 9), fg="darkgreen")
        self.camera_status_label.pack(pady=10)

    def load_model(self):
        """延迟加载模型"""
        if self.model is None:
            self.update_status("正在加载模型...")
            try:
                self.model = YOLO(r'D:\Code\YOLO_8Cell\runs\classify_1\weights\best.pt')
                self.update_status("模型加载完成")
                return True
            except Exception as e:
                messagebox.showerror("错误", f"模型加载失败: {e}")
                return False
        return True

    def get_class_name_display(self, class_name):
        """获取类别的中英文显示名称"""
        chinese_name = self.cell_classes.get(class_name, class_name)
        return f"{class_name}\n({chinese_name})"

    def update_status(self, text):
        """更新状态栏"""
        self.safe_ui_update("_update_status_text", text)

    def _update_status_text(self, text):
        """实际更新状态栏文本（在主线程执行）"""
        if self.root.winfo_exists():
            self.status_label.config(text=text)

    def predict_image(self):
        """图片识别功能"""
        self.current_mode = 'image'
        self.setup_image_mode()

        if not self.load_model():
            return

        file_path = filedialog.askopenfilename(
            title="选择细胞图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            try:
                self.update_file_label(f"图片文件: {os.path.basename(file_path)}")
                self.update_status("识别中...")

                # ====== 清空并准备显示结果 ======
                self.clear_result_text()
                self.append_result_text("🔍 开始识别...\n")
                self.append_result_text(f"文件: {file_path}\n\n")

                # ====== 在新线程中进行预测 ======
                thread = threading.Thread(target=self._predict_image_thread, args=(file_path,))
                thread.daemon = True
                thread.start()

            except Exception as e:
                self.append_result_text(f"❌ 识别失败: {str(e)}\n")
                self.update_status("识别失败")
                messagebox.showerror("错误", f"识别失败: {e}")

    def _predict_image_thread(self, file_path):
        """图片识别的线程函数"""
        try:
            # ====== 进行预测 ======
            results = self.model.predict(source=file_path, imgsz=224)
            result = results[0]

            # 在UI线程中显示结果
            self.safe_ui_update("_show_detailed_results", result, file_path)
            self.update_status("识别完成")

        except Exception as e:
            error_msg = f"❌ 识别失败: {str(e)}\n"
            self.safe_ui_update("_append_result_text", error_msg)
            self.update_status("识别失败")

    def clear_result_text(self):
        """清空结果文本"""
        self.safe_ui_update("_clear_result_text")

    def _clear_result_text(self):
        """实际清空结果文本（在主线程执行）"""
        if hasattr(self, 'result_text') and self.result_text.winfo_exists():
            self.result_text.delete(1.0, tk.END)

    def append_result_text(self, text):
        """追加结果文本"""
        self.safe_ui_update("_append_result_text", text)

    def _append_result_text(self, text):
        """实际追加结果文本（在主线程执行）"""
        if hasattr(self, 'result_text') and self.result_text.winfo_exists():
            self.result_text.insert(tk.END, text)
            self.result_text.see(tk.END)

    def update_file_label(self, text):
        """更新文件标签"""
        self.safe_ui_update("_update_file_label", text)

    def _update_file_label(self, text):
        """实际更新文件标签（在主线程执行）"""
        if self.file_label.winfo_exists():
            self.file_label.config(text=text)

    def _show_detailed_results(self, result, file_path):
        """显示详细识别结果（在主线程执行）"""
        if not hasattr(self, 'result_text') or not self.result_text.winfo_exists():
            return

        # ====== 获取所有类别概率 ======
        class_probs = []
        for i, prob in enumerate(result.probs.data):
            class_name = result.names[i]
            confidence = prob.item()
            class_probs.append((class_name, confidence))

        # ====== 按置信度排序 ======
        class_probs.sort(key=lambda x: x[1], reverse=True)

        # ====== 构建结果文本 ======
        self.result_text.insert(tk.END, "\n" + "=" * 60 + "\n")
        self.result_text.insert(tk.END, "🎯 识别结果\n")
        self.result_text.insert(tk.END, "=" * 60 + "\n\n")

        # ====== 显示最高概率结果 ======
        top_class, top_conf = class_probs[0]
        display_name = self.get_class_name_display(top_class)
        self.result_text.insert(tk.END, f"🏆 最终分类: {display_name}\n")
        self.result_text.insert(tk.END, f"📊 置信度: {top_conf:.4f} ({top_conf * 100:.2f}%)\n\n")

        # ====== 显示所有类别概率 ======
        self.result_text.insert(tk.END, "📈 详细概率分布:\n")
        self.result_text.insert(tk.END, "-" * 40 + "\n")
        for i, (class_name, conf) in enumerate(class_probs, 1):
            if conf > 0.0001:  # 只显示有显著概率的类别
                percentage = conf * 100
                bar = "█" * int(percentage / 5)  # 简单进度条
                display_name = self.get_class_name_display(class_name)
                self.result_text.insert(tk.END, f"{i:2d}. {display_name:<25}: {conf:.4f} {bar} ({percentage:5.2f}%)\n")

        self.result_text.insert(tk.END, "\n" + "=" * 60 + "\n")

        # ====== 性能信息 ======
        if hasattr(result, 'speed'):
            speed_info = result.speed
            total_time = speed_info.get('preprocess', 0) + speed_info.get('inference', 0) + speed_info.get(
                'postprocess', 0)
            self.result_text.insert(tk.END, f"⏱️  处理时间: {total_time:.1f}ms\n")
            self.result_text.insert(tk.END, f"   - 预处理: {speed_info.get('preprocess', 0):.1f}ms\n")
            self.result_text.insert(tk.END, f"   - 推理: {speed_info.get('inference', 0):.1f}ms\n")
            self.result_text.insert(tk.END, f"   - 后处理: {speed_info.get('postprocess', 0):.1f}ms\n")

    def predict_video(self):
        """视频识别功能"""
        self.current_mode = 'video'
        self.setup_video_mode()

        if not self.load_model():
            return

        file_path = filedialog.askopenfilename(
            title="选择细胞视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )

        if file_path:
            # ====== 重置视频控制状态 ======
            self.video_processing = False
            self.safe_ui_update("_update_video_controls", "loaded")
            self.video_path = file_path
            self.update_file_label(f"视频文件: {os.path.basename(file_path)}")

            # 清空实时结果显示
            self.clear_realtime_results()
            self.update_status("视频已加载，点击播放开始识别")

    def _update_video_controls(self, state):
        """更新视频控制按钮状态（在主线程执行）"""
        if not hasattr(self, 'btn_play') or not self.btn_play.winfo_exists():
            return

        if state == "loaded":
            self.btn_play.config(state=tk.NORMAL)
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.DISABLED)
        elif state == "playing":
            self.btn_play.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.NORMAL)
        elif state == "paused":
            self.btn_play.config(state=tk.NORMAL)
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
        elif state == "stopped":
            self.btn_play.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.DISABLED)

    def clear_realtime_results(self):
        """清空实时结果显示"""
        self.safe_ui_update("_clear_realtime_results")

    def _clear_realtime_results(self):
        """实际清空实时结果显示（在主线程执行）"""
        if not hasattr(self, 'main_class_label') or not self.main_class_label.winfo_exists():
            return

        self.main_class_label.config(text="无", fg="red")
        self.confidence_label.config(text="置信度: --")
        self.progress_label.config(text="状态: 等待开始")

        if hasattr(self, 'detail_text') and self.detail_text.winfo_exists():
            self.detail_text.config(state=tk.NORMAL)
            self.detail_text.delete(1.0, tk.END)
            self.detail_text.insert(tk.END, "视频加载完成\n点击播放开始识别")
            self.detail_text.config(state=tk.DISABLED)

    def play_video(self):
        """播放视频并进行识别"""
        if hasattr(self, 'video_path'):
            self.video_processing = True
            self.safe_ui_update("_update_video_controls", "playing")

            # ====== 清空实时结果 ======
            self.clear_realtime_results()
            self.safe_ui_update("_prepare_video_details")

            # ====== 在新线程中处理视频 ======
            thread = threading.Thread(target=self.process_video)
            thread.daemon = True
            thread.start()

    def _prepare_video_details(self):
        """准备视频详细信息显示（在主线程执行）"""
        if hasattr(self, 'detail_text') and self.detail_text.winfo_exists():
            self.detail_text.config(state=tk.NORMAL)
            self.detail_text.delete(1.0, tk.END)
            self.detail_text.insert(tk.END, "开始识别...\n")
            self.detail_text.config(state=tk.DISABLED)

    def pause_video(self):
        """暂停视频"""
        self.video_processing = False
        self.safe_ui_update("_update_video_controls", "paused")

    def stop_video(self):
        """停止视频"""
        self.video_processing = False
        self.safe_ui_update("_update_video_controls", "stopped")
        self.safe_ui_update("_reset_video_display")
        self.update_status("视频识别已停止")

        # 释放视频捕获对象
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None

    def _reset_video_display(self):
        """重置视频显示（在主线程执行）"""
        if hasattr(self, 'video_label') and self.video_label.winfo_exists():
            self.video_label.config(image='', text="视频预览区域\n\n选择视频文件后点击播放")

    def process_video(self):
        """处理视频帧"""
        try:
            self.video_cap = cv2.VideoCapture(self.video_path)
            if not self.video_cap.isOpened():
                self.safe_ui_update("_video_error", "无法打开视频文件")
                return

            # 获取视频信息
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 如果fps为0（图片合成的视频可能出现这种情况），设置默认fps
            if fps <= 0:
                fps = 25  # 默认25fps

            print(f"视频信息: {frame_count}帧, {fps}fps")

            frame_idx = 0
            analysis_interval = max(1, int(fps / 5))  # 每秒分析5次

            while self.video_cap.isOpened() and self.video_processing:
                ret, frame = self.video_cap.read()
                if not ret:
                    break

                frame_idx += 1

                # ====== 显示当前帧 ======
                self.safe_ui_update("_display_frame", frame, self.video_label)

                # ====== 定期进行分析 ======
                if frame_idx % analysis_interval == 0:
                    try:
                        results = self.model.predict(source=frame, imgsz=224, verbose=False)
                        result = results[0]
                        # 通过队列更新结果
                        self.safe_ui_update("_update_realtime_results", result, frame_idx, frame_count)
                    except Exception as e:
                        print(f"视频帧分析错误: {e}")

                # ====== 控制播放速度 ======
                delay = 1.0 / fps if fps > 0 else 0.04
                time.sleep(delay)

            self.video_cap.release()
            self.video_cap = None

            if frame_idx > 0:
                self.safe_ui_update("_video_completed")

        except Exception as e:
            self.safe_ui_update("_video_error", str(e))

    def _display_frame(self, frame, label_widget):
        """在GUI中显示视频帧（在主线程执行）"""
        try:
            if not label_widget or not label_widget.winfo_exists():
                return

            # ====== 调整帧大小以适应显示区域 ======
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # 计算缩放比例，保持宽高比
            display_w, display_h = 480, 360
            if w > 0 and h > 0:
                scale = min(display_w / w, display_h / h)
                new_w, new_h = int(w * scale), int(h * scale)

                if new_w > 0 and new_h > 0:
                    frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                    img = Image.fromarray(frame_resized)
                    imgtk = ImageTk.PhotoImage(image=img)

                    # 保持引用防止垃圾回收
                    if hasattr(label_widget, '_image_ref'):
                        label_widget._image_ref = imgtk
                    else:
                        label_widget._image_ref = imgtk

                    label_widget.config(image=imgtk, text="")

        except Exception as e:
            print(f"显示帧错误: {e}")

    def _update_realtime_results(self, result, frame_idx, total_frames):
        """更新实时识别结果（在主线程执行）"""
        if not hasattr(self, 'main_class_label') or not self.main_class_label.winfo_exists():
            return

        if hasattr(result, 'probs') and result.probs is not None:
            # ====== 获取最高概率的类别 ======
            top_class_idx = result.probs.top1
            top_confidence = result.probs.top1conf.item()
            class_name = result.names[top_class_idx]

            # ====== 设置高置信度阈值 - 只有大于95%才显示具体类别 ======
            confidence_threshold = 0.95

            if top_confidence < confidence_threshold:
                # 置信度太低，显示"无"
                display_class = "无"
                confidence_text = f"置信度: {top_confidence:.3f} (低于95%)"
                color = "red"
            else:
                display_class = self.get_class_name_display(class_name)
                confidence_text = f"置信度: {top_confidence:.3f}"
                color = "green"  # 95%以上都显示绿色

            # ====== 更新主要类别显示 ======
            self.main_class_label.config(text=display_class, fg=color)
            self.confidence_label.config(text=confidence_text)

            # ====== 更新进度 ======
            if total_frames > 0:
                progress = (frame_idx / total_frames) * 100
                self.progress_label.config(text=f"进度: {progress:.1f}% ({frame_idx}/{total_frames})")
            else:
                self.progress_label.config(text=f"处理中: 第{frame_idx}帧")

            # ====== 更新详细信息（只显示当前帧的结果） ======
            if hasattr(self, 'detail_text') and self.detail_text.winfo_exists():
                self.detail_text.config(state=tk.NORMAL)
                self.detail_text.delete(1.0, tk.END)

                # 获取所有类别概率（只显示大于1%的）
                class_probs = []
                for i, prob in enumerate(result.probs.data):
                    class_name_item = result.names[i]
                    confidence = prob.item()
                    if confidence > 0.01:  # 只显示概率大于1%的类别
                        class_probs.append((class_name_item, confidence))

                # 按置信度排序
                class_probs.sort(key=lambda x: x[1], reverse=True)

                # 显示详细信息
                self.detail_text.insert(tk.END, f"帧: {frame_idx}\n")
                self.detail_text.insert(tk.END, f"识别结果:\n\n")

                if top_confidence < confidence_threshold:
                    self.detail_text.insert(tk.END, f"未检测到明显的细胞类别\n")
                    self.detail_text.insert(tk.END, f"最高概率: {top_confidence * 100:.1f}%\n")
                    self.detail_text.insert(tk.END, f"(需要 >95% 才显示类别)\n")
                else:
                    self.detail_text.insert(tk.END, f"✅ 检测到: {self.get_class_name_display(class_name)}\n\n")
                    for i, (cls_name, conf) in enumerate(class_probs[:3]):  # 只显示前3个
                        percentage = conf * 100
                        if percentage > 1:  # 只显示大于1%的
                            bar = "█" * int(percentage / 10)  # 简化进度条
                            display_name = self.get_class_name_display(cls_name)
                            self.detail_text.insert(tk.END, f"{display_name}: {percentage:.1f}% {bar}\n")

                self.detail_text.config(state=tk.DISABLED)

            # ====== 更新状态栏 ======
            if top_confidence < confidence_threshold:
                status_text = f"识别中... {frame_idx}帧 - 未检测到明显类别"
            else:
                status_text = f"识别中... {frame_idx}帧 - 检测到: {self.cell_classes.get(class_name, class_name)}"

            if total_frames > 0:
                status_text += f" ({frame_idx}/{total_frames})"

            self.status_label.config(text=status_text)

    def _video_completed(self):
        """视频处理完成（在主线程执行）"""
        if not hasattr(self, 'main_class_label') or not self.main_class_label.winfo_exists():
            return

        self.main_class_label.config(text="识别完成", fg="green")
        self.progress_label.config(text="进度: 100% - 完成")
        self.status_label.config(text="视频识别完成")
        self.safe_ui_update("_update_video_controls", "stopped")

        if hasattr(self, 'detail_text') and self.detail_text.winfo_exists():
            self.detail_text.config(state=tk.NORMAL)
            self.detail_text.insert(tk.END, "\n\n✅ 视频识别完成！")
            self.detail_text.config(state=tk.DISABLED)

    def _video_error(self, error_msg):
        """视频处理错误（在主线程执行）"""
        messagebox.showerror("错误", f"视频处理失败: {error_msg}")
        self.status_label.config(text="视频处理失败")
        self.stop_video()

    # 摄像头相关方法保持不变...
    def predict_camera(self):
        """摄像头识别功能"""
        self.current_mode = 'camera'
        self.setup_camera_mode()
        self.update_file_label("摄像头识别模式")

        if not self.load_model():
            return

        self.update_status("摄像头模式就绪，点击开始进行实时识别")

    def start_camera(self):
        """开始摄像头识别"""
        try:
            # ====== 尝试打开摄像头 ======
            self.cap = cv2.VideoCapture(0)  # 0表示默认摄像头
            if not self.cap.isOpened():
                messagebox.showerror("错误", "无法打开摄像头，请检查摄像头连接")
                return

            self.camera_processing = True
            self.safe_ui_update("_update_camera_controls", "started")

            # ====== 清空摄像头结果显示 ======
            self.clear_camera_results()
            self.safe_ui_update("_prepare_camera_details")

            # ====== 在新线程中处理摄像头 ======
            thread = threading.Thread(target=self.process_camera)
            thread.daemon = True
            thread.start()

        except Exception as e:
            messagebox.showerror("错误", f"启动摄像头失败: {e}")

    def _update_camera_controls(self, state):
        """更新摄像头控制按钮状态（在主线程执行）"""
        if not hasattr(self, 'btn_camera_start') or not self.btn_camera_start.winfo_exists():
            return

        if state == "started":
            self.btn_camera_start.config(state=tk.DISABLED)
            self.btn_camera_stop.config(state=tk.NORMAL)
        elif state == "stopped":
            self.btn_camera_start.config(state=tk.NORMAL)
            self.btn_camera_stop.config(state=tk.DISABLED)

    def stop_camera(self):
        """停止摄像头识别"""
        self.camera_processing = False
        self.safe_ui_update("_update_camera_controls", "stopped")

        if self.cap:
            self.cap.release()
            self.cap = None

        # 清空摄像头显示
        self.safe_ui_update("_reset_camera_display")
        self.update_status("摄像头识别已停止")

    def _reset_camera_display(self):
        """重置摄像头显示（在主线程执行）"""
        if hasattr(self, 'camera_label') and self.camera_label.winfo_exists():
            self.camera_label.config(image='', text="摄像头预览区域\n\n点击开始进行实时识别")

    def clear_camera_results(self):
        """清空摄像头结果显示"""
        self.safe_ui_update("_clear_camera_results")

    def _clear_camera_results(self):
        """实际清空摄像头结果显示（在主线程执行）"""
        if not hasattr(self, 'camera_main_class_label') or not self.camera_main_class_label.winfo_exists():
            return

        self.camera_main_class_label.config(text="无", fg="red")
        self.camera_confidence_label.config(text="置信度: --")
        self.camera_status_label.config(text="状态: 等待开始")

    def _prepare_camera_details(self):
        """准备摄像头详细信息显示（在主线程执行）"""
        if hasattr(self, 'camera_detail_text') and self.camera_detail_text.winfo_exists():
            self.camera_detail_text.config(state=tk.NORMAL)
            self.camera_detail_text.delete(1.0, tk.END)
            self.camera_detail_text.insert(tk.END, "摄像头已启动\n开始实时识别...\n")
            self.camera_detail_text.config(state=tk.DISABLED)

    def process_camera(self):
        """处理摄像头帧"""
        frame_count = 0
        analysis_interval = 5  # 每5帧分析一次，提高实时性

        while self.camera_processing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            # ====== 显示当前帧 ======
            self.safe_ui_update("_display_frame", frame, self.camera_label)

            # ====== 定期进行分析 ======
            if frame_count % analysis_interval == 0:
                try:
                    results = self.model.predict(source=frame, imgsz=224, verbose=False)
                    result = results[0]
                    # 通过队列更新结果
                    self.safe_ui_update("_update_camera_results", result, frame_count)
                except Exception as e:
                    print(f"摄像头帧分析错误: {e}")

            # ====== 控制处理速度 ======
            time.sleep(0.03)  # 约30fps

        if self.cap:
            self.cap.release()
            self.cap = None

    def _update_camera_results(self, result, frame_count):
        """更新摄像头识别结果（在主线程执行）"""
        if not hasattr(self, 'camera_main_class_label') or not self.camera_main_class_label.winfo_exists():
            return

        if hasattr(result, 'probs') and result.probs is not None:
            # ====== 获取最高概率的类别 ======
            top_class_idx = result.probs.top1
            top_confidence = result.probs.top1conf.item()
            class_name = result.names[top_class_idx]

            # ====== 设置高置信度阈值 - 只有大于95%才显示具体类别 ======
            confidence_threshold = 0.95

            if top_confidence < confidence_threshold:
                # 置信度太低，显示"无"
                display_class = "无"
                confidence_text = f"置信度: {top_confidence:.3f} (低于95%)"
                color = "red"
            else:
                display_class = self.get_class_name_display(class_name)
                confidence_text = f"置信度: {top_confidence:.3f}"
                color = "green"  # 95%以上都显示绿色

            # ====== 更新主要类别显示 ======
            self.camera_main_class_label.config(text=display_class, fg=color)
            self.camera_confidence_label.config(text=confidence_text)

            # ====== 更新状态信息 ======
            self.camera_status_label.config(text=f"已处理帧数: {frame_count}")

            # ====== 更新详细信息（只显示当前帧的结果） ======
            if hasattr(self, 'camera_detail_text') and self.camera_detail_text.winfo_exists():
                self.camera_detail_text.config(state=tk.NORMAL)
                self.camera_detail_text.delete(1.0, tk.END)

                # 获取所有类别概率（只显示大于1%的）
                class_probs = []
                for i, prob in enumerate(result.probs.data):
                    class_name_item = result.names[i]
                    confidence = prob.item()
                    if confidence > 0.01:  # 只显示概率大于1%的类别
                        class_probs.append((class_name_item, confidence))

                # 按置信度排序
                class_probs.sort(key=lambda x: x[1], reverse=True)

                # 显示详细信息
                self.camera_detail_text.insert(tk.END, f"帧: {frame_count}\n")
                self.camera_detail_text.insert(tk.END, f"实时识别结果:\n\n")

                if top_confidence < confidence_threshold:
                    self.camera_detail_text.insert(tk.END, f"未检测到明显的细胞类别\n")
                    self.camera_detail_text.insert(tk.END, f"最高概率: {top_confidence * 100:.1f}%\n")
                    self.camera_detail_text.insert(tk.END, f"(需要 >95% 才显示类别)\n")
                else:
                    self.camera_detail_text.insert(tk.END, f"✅ 检测到: {self.get_class_name_display(class_name)}\n\n")
                    for i, (cls_name, conf) in enumerate(class_probs[:3]):  # 只显示前3个
                        percentage = conf * 100
                        if percentage > 1:  # 只显示大于1%的
                            bar = "█" * int(percentage / 10)  # 简化进度条
                            display_name = self.get_class_name_display(cls_name)
                            self.camera_detail_text.insert(tk.END, f"{display_name}: {percentage:.1f}% {bar}\n")

                self.camera_detail_text.config(state=tk.DISABLED)

            # ====== 更新状态栏 ======
            if top_confidence < confidence_threshold:
                self.status_label.config(text=f"摄像头识别中... 已处理{frame_count}帧 - 未检测到明显类别")
            else:
                self.status_label.config(
                    text=f"摄像头识别中... 已处理{frame_count}帧 - 检测到: {self.cell_classes.get(class_name, class_name)}")

    def run(self):
        """运行应用"""
        self.root.mainloop()


if __name__ == "__main__":
    print("启动细胞分类系统...")
    app = SimpleCellClassifier()
    app.run()