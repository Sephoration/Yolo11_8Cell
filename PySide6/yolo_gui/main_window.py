# yolo_gui/main_window.py
"""
YOLO11-PySide6-GUI 主窗口

架构说明：
- 负责界面显示、按钮交互、参数设置
- 具体目标检测逻辑封装在 yolo_gui/Yolo11Detect.py 的 YOLO11Detector 类中
"""

import os
import time
import cv2
import  numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QComboBox,
    QSlider, QProgressBar, QSpacerItem, QSizePolicy,
    QFileDialog
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt,QCoreApplication

from yolo_gui.YoloTracker import yoloTracker

# from yolo_gui.Yolo11Detect import YOLO11Detector

class YOLOMainWindow(QMainWindow):
    """
    YOLO11 可视化主窗口

    分区说明：
    1. 初始化区：窗口标题、成员变量、调用 _init_ui()
    2. UI 构建区：只负责布局和控件创建，不写业务逻辑
    3. 行为函数区：响应按钮 / 下拉框事件（Image、Toggle View 等）
    4. 工具函数区：图像显示、模型列表加载等通用小工具
    """
    # ======================================================================
    # 1. 初始化区（Initialization）
    # ======================================================================
    def __init__(self):
        super().__init__()

        # 窗口基础设置
        self.setWindowTitle("东莞理工 - 粤台产业科技学院 - 计算机视觉研究中心")
        self.resize(1200, 700)

        # 视图模式：False = 双视图；True = 只显示右侧视图
        self.single_view = False

        # 一些成员变量占位（方便将来扩展）
        self.status_label: QLabel | None = None
        self.progress_bar: QProgressBar | None = None
        self.view_left: QLabel | None = None
        self.view_right: QLabel | None = None
        self.combo_model: QComboBox | None = None
        self.slider_iou: QSlider | None = None
        self.slider_conf: QSlider | None = None
        self.slider_delay: QSlider | None = None
        self.slider_line_width: QSlider | None = None
        self.card_model: QFrame | None = None

        # 初始化界面
        self._init_ui()

    # ======================================================================
    # 2. UI 构建区（UI Setup：布局 + 控件创建）
    # ======================================================================
    def _init_ui(self):
        """构建整体三段式结构：顶部 / 中间 / 底部。"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 整体垂直布局：上 / 中 / 下 三部分
        main_v_layout = QVBoxLayout(central_widget)
        main_v_layout.setContentsMargins(5, 5, 5, 5)
        main_v_layout.setSpacing(5)

        # ---------- 2.1 顶部区域 Top Bar ----------
        top_bar = QWidget()
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(0, 0, 0, 0)
        top_bar_layout.setSpacing(5)

        self.card_classes = self._create_top_card("Classes\n--")
        self.card_targets = self._create_top_card("Targets\n--")
        self.card_fps = self._create_top_card("FPS\n--")
        self.card_model = self._create_top_card("Model\n(yolo11n.pt)")

        top_bar_layout.addWidget(self.card_classes)
        top_bar_layout.addWidget(self.card_targets)
        top_bar_layout.addWidget(self.card_fps)
        top_bar_layout.addWidget(self.card_model)

        main_v_layout.addWidget(top_bar)

        # ---------- 2.2 中间区域 Center Area ----------
        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(5)

        # 左侧工具栏
        left_toolbar = self._create_left_toolbar()
        self._create_left_toolbar()
        center_layout.addWidget(left_toolbar)


        # 中间视图区
        middle_panel = self._create_middle_panel()
        center_layout.addWidget(middle_panel, stretch=1)

        # 右侧设置栏
        right_settings = self._create_right_settings_panel()
        center_layout.addWidget(right_settings)

        main_v_layout.addWidget(center_widget, stretch=1)

        # ---------- 2.3 底部区域 Bottom Bar ----------
        bottom_bar = QWidget()
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(5)

        self.status_label = QLabel("当前为 GUI + Image 检测示例版本")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        bottom_layout.addWidget(self.status_label)
        bottom_layout.addWidget(self.progress_bar, stretch=1)

        main_v_layout.addWidget(bottom_bar)

    # ---------- 2.x 顶部信息卡 ----------
    def _create_top_card(self, text: str) -> QFrame:
        """顶部信息卡片：Classes / Targets / FPS / Model。"""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)

        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        return frame

    # ---------- 2.x 左侧工具栏 ----------
    def _create_left_toolbar(self) -> QFrame:
        """
        左侧工具栏：
            - Home / Image / Video / Camera / Folder / Toggle View / Settings
        当前实现：
            - Image：打开图像并进行目标检测
            - Toggle View：切换双视图 / 单视图（纯 GUI 行为）
        """
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        btn_home = QPushButton("Home")
        btn_image = QPushButton("Image")
        btn_folder = QPushButton("Folder")
        btn_video = QPushButton("Video")
        btn_camera = QPushButton("Camera")
        btn_toggle_view = QPushButton("Toggle View")
        btn_settings = QPushButton("Settings")

        # 目前只实现 Home 与 Toggle View 的行为
        btn_home.clicked.connect(self.on_home_clicked)
        btn_image = QPushButton("Image")
        btn_toggle_view.clicked.connect(self.toggle_view_mode)
        btn_image.clicked.connect(self.open_image_dialog)


        for btn in [
            btn_home, btn_image, btn_folder, btn_video, btn_camera,
            btn_toggle_view, btn_settings
        ]:
            btn.setMinimumHeight(30)
            layout.addWidget(btn)

        layout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        return frame

    # ---------- 2.x 中间视图区 ----------
    def _create_middle_panel(self) -> QFrame:
        """
        中间视图区：
            - 左视图：Left View（原始图像）
            - 右视图：Right View（检测结果）
        """
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        self.view_left = QLabel("Left View\n(原始图像)")
        self.view_left.setAlignment(Qt.AlignCenter)
        self.view_left.setStyleSheet("background-color: #f0f0f0;")

        self.view_right = QLabel("Right View\n(检测结果)")
        self.view_right.setAlignment(Qt.AlignCenter)
        self.view_right.setStyleSheet("background-color: #f0f0f0;")

        layout.addWidget(self.view_left, stretch=1)
        layout.addWidget(self.view_right, stretch=1)

        return frame

    # ---------- 2.x 右侧设置栏 ----------
    def _create_right_settings_panel(self) -> QFrame:
        """
        右侧设置栏：
            - Model 下拉框（自动扫描 models/ 目录）
            - IOU / Confidence / Delay / LineWidth 滑条
            - 底部按钮（占位：打开图片 / 检测 / 保存结果）
        """
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # --- Model 下拉选择 ---
        label_model = QLabel("Model")
        self.combo_model = QComboBox()

        model_files = self._load_model_list()
        self.combo_model.addItems(model_files)

        # 若存在 yolo11n.pt 则优先选中它
        target_name = "yolo11n.pt"
        for index, path in enumerate(model_files):
            if os.path.basename(path).lower() == target_name:
                self.combo_model.setCurrentIndex(index)
                break

        # 当模型选择变化时，更新顶部 Model 卡片显示
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)

        layout.addWidget(label_model)
        layout.addWidget(self.combo_model)

        # --- IOU Slider ---
        label_iou = QLabel("IOU Threshold")
        self.slider_iou = QSlider(Qt.Horizontal)
        self.slider_iou.setRange(0, 100)
        self.slider_iou.setValue(50)
        layout.addWidget(label_iou)
        layout.addWidget(self.slider_iou)

        # --- Confidence Slider ---
        label_conf = QLabel("Confidence Threshold")
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(0, 100)
        self.slider_conf.setValue(50)
        layout.addWidget(label_conf)
        layout.addWidget(self.slider_conf)

        # --- Delay Slider ---
        label_delay = QLabel("Delay (ms)")
        self.slider_delay = QSlider(Qt.Horizontal)
        self.slider_delay.setRange(0, 200)
        self.slider_delay.setValue(0)
        layout.addWidget(label_delay)
        layout.addWidget(self.slider_delay)

        # --- Line Width Slider ---
        label_line_width = QLabel("Line Width")
        self.slider_line_width = QSlider(Qt.Horizontal)
        self.slider_line_width.setRange(1, 10)
        self.slider_line_width.setValue(3)
        layout.addWidget(label_line_width)
        layout.addWidget(self.slider_line_width)

        # --- 底部按钮（占位） ---
        self.btn_open = QPushButton("打开图片（右侧按钮，占位）")
        self.btn_detect = QPushButton("检测（待实现）")
        btn_save_result = QPushButton("保存结果（待实现）")

        layout.addWidget(self.btn_open)
        layout.addWidget(self.btn_detect)
        layout.addWidget(btn_save_result)

        layout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        return frame

    # ======================================================================
    # 3. 行为 / 功能函数区（Behavior / Feature Functions）
    # ======================================================================
    def toggle_view_mode(self):
        """
        切换视图模式：
            - 双视图模式：左视图 + 右视图
            - 单视图模式：只显示右视图（左视图隐藏）
        """
        self.single_view = not self.single_view

        if self.single_view:
            if self.view_left:
                self.view_left.hide()
            if self.status_label:
                self.status_label.setText("视图模式：只显示右视图")
        else:
            if self.view_left:
                self.view_left.show()
            if self.status_label:
                self.status_label.setText("视图模式：左 / 右双视图")

    def on_home_clicked(self):
        """
        Home 按钮功能：
        将界面重置为初始状态（不做侦测）：
            - 左右视图恢复为占位文字
            - 视图模式恢复为双视图
            - 状态栏文字恢复
            - 进度条清零
            - 滑条恢复默认值
        """
        # 1. 恢复视图模式为“双视图”
        self.single_view = False
        if self.view_left:
            self.view_left.show()

        # 2. 左右视图恢复为占位状态
        if self.view_left:
            self.view_left.clear()
            self.view_left.setText("Left View\n(原始图像)")
            self.view_left.setAlignment(Qt.AlignCenter)

        if self.view_right:
            self.view_right.clear()
            self.view_right.setText("Right View\n(检测结果)")
            self.view_right.setAlignment(Qt.AlignCenter)

        # 3. 状态栏与进度条重置
        if self.status_label:
            self.status_label.setText("已重置为初始状态（Home）")

        if self.progress_bar:
            self.progress_bar.setValue(0)

        # 4. 滑条恢复默认值
        if self.slider_iou:
            self.slider_iou.setValue(50)
        if self.slider_conf:
            self.slider_conf.setValue(50)
        if self.slider_delay:
            self.slider_delay.setValue(0)
        if self.slider_line_width:
            self.slider_line_width.setValue(3)

    def _on_model_changed(self):
        """
        当用户在模型下拉框中选择不同模型时，
        自动更新顶部 Model 卡片中显示的模型名称。
        """
        if not (self.combo_model and self.card_model):
            return

        full_path = self.combo_model.currentText()
        filename = os.path.basename(full_path)

        # card_model 内部只有一个 QLabel
        label = self.card_model.findChild(QLabel)
        if label is not None:
            label.setText(f"Model\n{filename}")

    # ======================================================================
    # 4. 工具函数区（Utility Functions）
    # ======================================================================
    def _load_model_list(self):
        """
        扫描 models/ 目录，返回所有 .pt 模型完整路径列表。

        说明：
        - 供右侧 Model 下拉框使用
        - 若目录不存在或为空，返回空列表
        """
        model_dir = "../models"
        model_files: list[str] = []

        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.lower().endswith(".pt"):
                    model_files.append(os.path.join(model_dir, f))

        return model_files

    def open_image_dialog(self):
        """
        左侧工具栏「Image」按钮功能：
        1. 打开文件选择对话框
        2. 使用 YOLO11Detector 进行目标检测
        3. 左视图显示原图，右视图显示检测结果
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp)"
        )

        if not file_path:
            return  # 用户取消

        # 1. 初始化 YOLO11 检测器(使用当前选中的模型)
