"""
软件界面_code部分
可以单独导入图片、播放视频、使用摄像头
导入模型的时候会使用yolo_analyzer模块来分析是什么模型
按下开始时候才正式检测
"""

import os
import sys
import importlib
import traceback
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from PySide6.QtCore import QObject, QThread, Signal, QTimer, Qt, QMutex, QWaitCondition
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtWidgets import (
    QMessageBox, QFileDialog, QApplication, QDialog, QVBoxLayout, 
    QPushButton, QLabel, QScrollArea, QWidget
)

# 导入UI
from window_ui import YOLOMainWindowUI


class SimpleVideoPlayer(QObject):
    """极简视频播放器 - 只负责流畅显示，不涉及YOLO"""
    
    frame_ready = Signal(QImage)  # 帧就绪信号（用于显示）
    status_update = Signal(str)   # 状态更新
    progress_updated = Signal(int, int, float)  # 新增：进度更新信号 (当前帧, 总帧数, 当前时间)
    finished = Signal()           # 播放完成
    
    def __init__(self):
        super().__init__()
        self.playing = False
        self.cap = None
        self.current_frame = None  # 当前帧（numpy array）
        self.frame_mutex = threading.Lock()
        self.play_thread = None
        
        # 新增视频信息
        self.total_frames = 0
        self.current_frame_num = 0
        self.fps = 30.0
        self.duration = 0.0

        # 新增：用于 pause/resume 的事件（pause 时 clear -> 阻塞；resume 时 set -> 继续）
        self._pause_event = threading.Event()
        self._pause_event.set()   # 默认不阻塞
        self.paused = False

        # 优化开关：使用 grab()/retrieve() 模式读取可减少部分解码阻塞
        self._use_grab = True
    
    def play_video(self, video_path: str):
        """播放视频文件 - 极简版本"""
        if self.play_thread and self.play_thread.is_alive():
            self.stop()
        
        self.playing = True
        self.paused = False
        self._pause_event.set()
        self.play_thread = threading.Thread(
            target=self._video_playback_simple,
            args=(video_path,),
            daemon=True
        )
        self.play_thread.start()
    
    def play_camera(self, camera_id: int = 0):
        """播放摄像头 - 极简版本"""
        if self.play_thread and self.play_thread.is_alive():
            self.stop()
        
        self.playing = True
        self.paused = False
        self._pause_event.set()
        self.play_thread = threading.Thread(
            target=self._camera_playback_simple,
            args=(camera_id,),
            daemon=True
        )
        self.play_thread.start()
    
    def stop(self):
        """停止播放"""
        # 标记停止并释放等待，确保线程能退出
        self.playing = False
        self._pause_event.set()
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)
        
        self.finished.emit()
    
    def pause(self):
        """暂停播放（不会结束线程）"""
        # 只阻塞读取线程，不释放资源
        self.paused = True
        self._pause_event.clear()
    
    def resume(self):
        """继续播放（唤醒读取线程）"""
        # 如果线程已经结束，需要重新启动（尝试重启用于edge情形）
        if not (self.play_thread and self.play_thread.is_alive()) and self.cap is None and self.current_frame is None:
            # 线程已不存在且没有打开资源，无法自动恢复
            # 上层应在需要时重新调用 play_video/play_camera
            self.paused = False
            self._pause_event.set()
            return
        
        self.paused = False
        self._pause_event.set()
    
    def get_current_frame(self):
        """获取当前帧（用于抓取）"""
        with self.frame_mutex:
            # 返回拷贝，避免共享底层缓冲引起并发问题
            return self.current_frame.copy() if (self.current_frame is not None and hasattr(self.current_frame, 'copy')) else None

    def seek_frame(self, target_frame: int):
        """跳转到指定帧并立即读取一帧用于更新显示（供进度条拖动使用）"""
        try:
            import cv2
            if not self.cap or not self.cap.isOpened():
                return
            # 设置目标帧号并读取一帧
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(target_frame)))
            # 使用 grab/retrieve 尝试更快读取
            if self._use_grab:
                self.cap.grab()
                ret, frame = self.cap.retrieve()
            else:
                ret, frame = self.cap.read()
            if not ret:
                return
            # 拷贝帧
            with self.frame_mutex:
                self.current_frame = frame.copy()
                try:
                    self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                except:
                    pass
            # 发出进度与帧信号
            current_time = self.current_frame_num / max(1.0, self.fps)
            self.progress_updated.emit(self.current_frame_num, self.total_frames, current_time)
            # 转换为QImage并发送（显示用）
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            # 确保连续内存，避免QImage读共享内存出现问题
            frame_rgb = np.ascontiguousarray(frame_rgb)
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
            self.frame_ready.emit(q_img)
        except Exception:
            # 保持稳定，不打印额外日志
            traceback.print_exc()

    def _video_playback_simple(self, video_path: str):
        """简单的视频播放 - 专注于流畅显示"""
        try:
            import cv2
            import numpy as np

            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.status_update.emit(f"无法打开视频文件: {video_path}")
                return

            # 尝试设置较小的内部缓冲（部分 OpenCV 后端支持）
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            except:
                pass

            # 获取视频信息
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0

            if self.total_frames > 0:
                self.duration = self.total_frames / self.fps
            else:
                # 如果无法获取总帧数，设置为1000作为默认范围
                self.total_frames = 1000

            self.status_update.emit(f"开始播放视频: {os.path.basename(video_path)}")
            self.status_update.emit(f"总帧数: {self.total_frames}, FPS: {self.fps:.2f}")

            frame_interval = 1.0 / self.fps if self.fps > 0 else 0.033

            # 保证帧号有初始值
            try:
                self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            except:
                self.current_frame_num = 0

            # 如果使用 grab 模式，先做少量预抓取以减轻首帧延迟
            if self._use_grab:
                for _ in range(2):
                    try:
                        self.cap.grab()
                    except:
                        break

            while self.playing:
                # 如果被暂停，阻塞直到 resume（或 stop）
                if not self._pause_event.is_set():
                    self._pause_event.wait()
                    if not self.playing:
                        break

                loop_start = time.time()

                # 读取一帧：优先尝试 grab/retrieve（部分后端更快）
                if self._use_grab:
                    ok = self.cap.grab()
                    if not ok:
                        # 到尾或者读取失败，尝试回到0或退出
                        try:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.current_frame_num = 0
                            continue
                        except:
                            break
                    ret, frame = self.cap.retrieve()
                else:
                    ret, frame = self.cap.read()

                if not ret or frame is None:
                    # 视频到尾部，尝试循环
                    try:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_num = 0
                        continue
                    except:
                        break

                # 拷贝并保存当前帧（用于抓取）
                with self.frame_mutex:
                    self.current_frame = frame.copy()

                # 更新当前帧号（尽量从 cap 查询）
                try:
                    pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if pos >= 0:
                        self.current_frame_num = pos
                    else:
                        self.current_frame_num += 1
                except:
                    self.current_frame_num += 1

                # 发送进度信息
                current_time = self.current_frame_num / self.fps if self.fps > 0 else 0.0
                self.progress_updated.emit(self.current_frame_num, self.total_frames, current_time)

                # 转换为QImage并发送（显示用）
                try:
                    frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = np.ascontiguousarray(frame_rgb)
                    height, width, channel = frame_rgb.shape
                    bytes_per_line = 3 * width

                    # 创建QImage（基于连续内存），并拷贝确保数据独立
                    q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
                    self.frame_ready.emit(q_img)
                except Exception:
                    traceback.print_exc()

                # 更准确的等待：考虑本次解码耗时，使用 Event.wait 来支持 pause/即时恢复
                elapsed = time.time() - loop_start
                wait_time = max(0.0, frame_interval - elapsed)
                # 如果暂停，会在下次循环开始时阻塞
                self._pause_event.wait(timeout=wait_time)

            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None

        except Exception:
            self.status_update.emit(f"视频播放错误")
            traceback.print_exc()
        finally:
            self.playing = False
            self._pause_event.set()
            self.finished.emit()
    
    def _camera_playback_simple(self, camera_id: int):
        """简单的摄像头播放 - 专注于流畅显示"""
        try:
            import cv2
            import numpy as np

            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                self.status_update.emit(f"无法打开摄像头: {camera_id}")
                return

            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # 尝试设置较小缓冲
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except:
                pass

            # 摄像头启动时做少量预热读取，减少首帧卡顿
            for _ in range(3):
                try:
                    ret_warm, _ = self.cap.read()
                except:
                    ret_warm = False
                if not ret_warm:
                    break
                time.sleep(0.01)

            self.status_update.emit(f"开始摄像头实时显示")

            while self.playing:
                # 等待resume（如果处于暂停状态会阻塞在这里）
                if not self._pause_event.is_set():
                    self._pause_event.wait()
                    if not self.playing:
                        break

                loop_start = time.time()

                # 读取一帧
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.status_update.emit("无法读取摄像头画面")
                    time.sleep(0.02)
                    continue

                # 保存当前帧（用于抓取），做拷贝
                with self.frame_mutex:
                    self.current_frame = frame.copy()

                # 转换为QImage并发送（显示用）
                try:
                    frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = np.ascontiguousarray(frame_rgb)
                    height, width, channel = frame_rgb.shape
                    bytes_per_line = 3 * width

                    q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
                    self.frame_ready.emit(q_img)
                except Exception:
                    traceback.print_exc()

                # 控制帧率（约30fps），使用 wait 支持 pause 响应
                elapsed = time.time() - loop_start
                wait_time = max(0.0, 0.033 - elapsed)
                self._pause_event.wait(timeout=wait_time)

            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None

        except Exception:
            self.status_update.emit(f"摄像头播放错误")
            traceback.print_exc()
        finally:
            self.playing = False
            self._pause_event.set()
            self.finished.emit()


class FrameGrabberWorker(QObject):
    """帧抓取工作者 - 负责从播放器抓取帧并发送给YOLO"""
    
    frame_processed = Signal(QImage)  # 新增：处理后的帧（用于显示）
    frame_grabbed = Signal(object)    # 抓取的帧（numpy array）
    processing_complete = Signal(dict)  # 处理完成（统计信息）
    status_update = Signal(str)
    finished = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, video_player: SimpleVideoPlayer, yolo_module=None):
        super().__init__()
        self.video_player = video_player
        self.yolo_module = yolo_module
        self.processing = False
        self.grab_interval = 5  # 每5帧抓取一次
        self.frame_count = 0
        self.grab_thread = None
        
        # 处理统计
        self.total_frames_processed = 0
        self.total_detections = 0
        self.total_inference_time = 0.0
        self.start_time = 0
        
        # 性能跟踪
        self.last_frame_time = 0
    
    def start_grabbing(self, grab_interval: int = 5):
        """开始抓取帧"""
        if self.grab_thread and self.grab_thread.is_alive():
            self.stop_grabbing()
        
        self.grab_interval = grab_interval
        self.processing = True
        self.frame_count = 0
        self.total_frames_processed = 0
        self.total_detections = 0
        self.total_inference_time = 0.0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
        self.grab_thread = threading.Thread(
            target=self._grab_frames,
            daemon=True
        )
        self.grab_thread.start()
    
    def stop_grabbing(self):
        """停止抓取"""
        self.processing = False
        if self.grab_thread and self.grab_thread.is_alive():
            self.grab_thread.join(timeout=1.0)
        self.finished.emit()
    
    def set_yolo_module(self, yolo_module):
        """设置YOLO模块"""
        self.yolo_module = yolo_module
    
    def _grab_frames(self):
        """抓取帧的核心逻辑"""
        self.status_update.emit("开始抓取帧进行检测...")
        
        try:
            import cv2
            import numpy as np
            
            while self.processing:
                # 从播放器获取当前帧
                frame = self.video_player.get_current_frame()
                
                if frame is None:
                    time.sleep(0.1)  # 等待帧就绪
                    continue
                
                self.frame_count += 1
                
                # 按间隔抓取（避免每帧都处理）
                if self.frame_count % self.grab_interval == 0:
                    try:
                        current_time = time.time()
                        frame_interval = current_time - self.last_frame_time
                        self.last_frame_time = current_time
                        
                        # 如果有YOLO模块，进行处理
                        if self.yolo_module and hasattr(self.yolo_module, 'process_frame'):
                            # 记录推理开始时间
                            inference_start = time.time()
                            
                            # 调用YOLO模块处理帧（返回字典）
                            result_dict = self.yolo_module.process_frame(frame)
                            
                            # 记录推理时间
                            inference_time = time.time() - inference_start
                            self.total_inference_time += inference_time
                            self.total_frames_processed += 1
                            
                            # 从字典中提取图像和结果
                            processed_frame = result_dict.get('image', frame)
                            results_data = result_dict.get('stats', {})
                            
                            # 提取统计信息
                            stats = self._extract_statistics(results_data)
                            stats['inference_time'] = inference_time * 1000  # 转换为毫秒
                            stats['fps'] = 1.0 / frame_interval if frame_interval > 0 else 0
                            
                            # 更新累计检测数
                            detection_count = stats.get('detection_count', 0)
                            if 'detection_count' in stats:
                                self.total_detections += detection_count
                                stats['total_detections'] = self.total_detections
                            
                            # 发送处理后的帧用于显示
                            if isinstance(processed_frame, np.ndarray):
                                # 将OpenCV图像转换为QImage
                                if len(processed_frame.shape) == 2:  # 灰度图
                                    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
                                elif processed_frame.shape[2] == 4:  # RGBA
                                    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2RGB)
                                else:  # BGR
                                    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                
                                height, width, channel = processed_rgb.shape
                                bytes_per_line = 3 * width
                                
                                q_img = QImage(processed_rgb.data, width, height, 
                                              bytes_per_line, QImage.Format_RGB888)
                                self.frame_processed.emit(q_img)
                            
                            # 发送统计信息
                            stats['total_processed'] = self.total_frames_processed
                            stats['avg_inference_time'] = (self.total_inference_time / 
                                                         self.total_frames_processed * 1000 
                                                         if self.total_frames_processed > 0 else 0)
                            
                            # 如果有分类信息，添加到统计中
                            if 'class_name' in result_dict and result_dict['class_name'] != '未知':
                                stats['class_name'] = result_dict['class_name']
                                stats['confidence'] = result_dict.get('confidence', 0.0)
                            
                            self.processing_complete.emit(stats)
                        
                    except Exception as e:
                        self.error_occurred.emit(f"帧处理错误: {str(e)}")
                        traceback.print_exc()
                
                # 控制抓取频率（不要太快）
                time.sleep(0.05)  # 每秒最多20次抓取
        
        except Exception as e:
            self.error_occurred.emit(f"抓取过程错误: {str(e)}")
            traceback.print_exc()
        finally:
            self.processing = False
            self.finished.emit()
    
    def _extract_statistics(self, results) -> Dict[str, Any]:
        """从YOLO结果中提取统计信息"""
        stats = {
            'detection_count': 0,
            'inference_time': 0,
            'fps': 0.0,
            'avg_confidence': 0.0,
            'classes': {},
            'tracked_objects': 0
        }
        
        if results is None:
            return stats
        
        # 如果是字典格式的结果
        if isinstance(results, dict):
            stats.update({k: results.get(k, v) for k, v in stats.items()})
            
            # 如果有具体字段，使用它们
            if 'detection_count' in results:
                stats['detection_count'] = results['detection_count']
            if 'avg_confidence' in results:
                stats['avg_confidence'] = results['avg_confidence']
            if 'class_name' in results:
                stats['classes'] = {results['class_name']: 1}
        
        return stats


class YOLOMainWindowLogic(QObject):
    """主窗口逻辑控制器 - 简化版本"""
    
    def __init__(self, ui_window: YOLOMainWindowUI):
        super().__init__()
        self.ui = ui_window
        
        # 核心组件
        self.video_player = SimpleVideoPlayer()      # 极简播放器
        self.frame_grabber = FrameGrabberWorker(self.video_player)  # 帧抓取器
        
        # 状态变量
        self.current_yolo_module = None
        self.model_loaded = False
        self.model_path = None
        self.selected_module_type = None
        
        # 处理状态
        self.is_processing = False      # 是否正在YOLO处理
        self.is_playing = False         # 是否正在播放
        self.current_file = None
        self.current_mode = None        # 'image', 'video', 'camera'
        
        # 默认参数
        self.default_params = {
            'iou_threshold': 0.45,
            'confidence_threshold': 0.5,
            'delay_ms': 10,
            'line_width': 2
        }
        
        # 先初始化UI状态，再设置连接
        self._init_ui_state()
        self._setup_connections()
        
        print("YOLO逻辑控制器初始化完成 - 简化版本")
    
    def _init_ui_state(self):
        """初始化UI状态"""
        # 获取UI组件引用
        self.left_panel = self.ui.get_left_panel()
        self.right_panel = self.ui.get_right_panel()
        
        # 设置默认参数
        self.right_panel.set_parameters(**self.default_params)
        
        # 显示初始状态
        self.left_panel.clear_display()
        self.right_panel.update_model_info()
        
        # 设置控制按钮状态
        self.right_panel.set_control_state(False)
    
    def _setup_connections(self):
        """设置信号连接"""
        # ===== 连接视频播放器信号 =====
        self.video_player.frame_ready.connect(self._on_player_frame)
        self.video_player.status_update.connect(self._on_status_update)
        self.video_player.progress_updated.connect(self._on_progress_updated)  # 新增进度信号
        self.video_player.finished.connect(self._on_player_finished)
        
        # ===== 连接帧抓取器信号 =====
        self.frame_grabber.frame_processed.connect(self._on_frame_processed)  # 新增：处理后的帧
        self.frame_grabber.frame_grabbed.connect(self._on_frame_grabbed)
        self.frame_grabber.processing_complete.connect(self._on_processing_complete)
        self.frame_grabber.status_update.connect(self._on_status_update)
        self.frame_grabber.error_occurred.connect(self._on_grabber_error)
        self.frame_grabber.finished.connect(self._on_grabber_finished)
        
        # ===== 文件菜单信号 =====
        self.ui.file_menu_init.connect(self._on_file_init)
        self.ui.file_menu_exit.connect(self._on_file_exit)
        
        # ===== 帮助菜单信号 =====
        self.ui.help_menu_about.connect(self._on_help_about)
        self.ui.help_menu_manual.connect(self._on_help_manual)
        
        # ===== 主要功能信号 =====
        self.ui.model_load.connect(self._on_model_load)
        self.ui.image_open.connect(self._on_image_open)
        self.ui.video_open.connect(self._on_video_open)
        self.ui.camera_open.connect(self._on_camera_open)
        
        # ===== 控制按钮信号 =====
        self.right_panel.start_inference.connect(self._on_start_inference)
        self.right_panel.stop_inference.connect(self._on_stop_inference)
        self.right_panel.save_screenshot.connect(self._on_save_screenshot)
        
        # ===== 左侧面板播放/暂停信号 =====
        self.ui.left_panel_play_pause.connect(self._on_play_pause_clicked)
        
        # ===== 左侧面板进度条信号 =====
        self.left_panel.progress_changed.connect(self._on_progress_changed)
    
    # ============================================================================
    # 信号处理方法
    # ============================================================================
    
    def _on_player_frame(self, q_image: QImage):
        """接收到播放器的原始帧 - 直接显示（无YOLO处理时）"""
        try:
            if not self.is_processing:
                pixmap = QPixmap.fromImage(q_image)
                self.left_panel.set_display_image(pixmap)
        except Exception as e:
            print(f"显示原始帧失败: {e}")
    
    def _on_frame_processed(self, q_image: QImage):
        """接收到处理后的帧 - 显示YOLO检测结果"""
        try:
            pixmap = QPixmap.fromImage(q_image)
            self.left_panel.set_display_image(pixmap)
        except Exception as e:
            print(f"显示处理帧失败: {e}")
    
    def _on_player_finished(self):
        """播放器完成"""
        self.is_playing = False
        self.left_panel.set_play_state(False)
        print("播放器停止")
    
    def _on_frame_grabbed(self, frame):
        """接收到抓取的帧 - 可以在这里处理或发送给其他模块"""
        # 这里可以保存帧、记录日志等
        pass
    
    def _on_processing_complete(self, stats: dict):
        """处理完成 - 更新统计信息"""
        try:
            # 更新基本统计信息
            self.right_panel.update_statistics(
                detection_count=stats.get('detection_count', 0),
                confidence=stats.get('avg_confidence', 0.0),
                inference_time=stats.get('inference_time', 0),
                fps=stats.get('fps', 0.0)
            )
            
            # 如果有类分布信息，更新详细统计
            if 'classes' in stats and stats['classes']:
                class_distribution = "\n".join([f"{cls}: {count}" for cls, count in stats['classes'].items()])
                self.right_panel.update_detailed_stats(
                    total_processed=stats.get('total_processed', 0),
                    total_detections=stats.get('total_detections', 0),
                    avg_inference_time=stats.get('avg_inference_time', 0),
                    class_distribution=class_distribution
                )
            
        except Exception as e:
            print(f"更新统计信息失败: {e}")
    
    def _on_grabber_error(self, error_msg: str):
        """抓取器错误"""
        print(f"抓取器错误: {error_msg}")
    
    def _on_grabber_finished(self):
        """抓取器完成"""
        self.is_processing = False
        self.right_panel.set_control_state(False)
        print("帧抓取停止")
    
    def _on_status_update(self, status: str):
        """状态更新"""
        print(f"状态: {status}")
    
    def _on_progress_updated(self, current_frame, total_frames, current_time):
        """视频进度更新"""
        try:
            if self.current_mode == 'video':
                # 更新进度条范围（使用0-1000范围，方便UI显示）
                self.left_panel.set_progress_range(0, 1000)
                
                # 计算进度值（0-1000）
                if total_frames > 0:
                    progress_value = int((current_frame / total_frames) * 1000)
                    self.left_panel.set_progress_value(progress_value)
                
                # 更新时间显示
                current_time_str = self._format_time(current_time)
                total_time_str = self._format_time(total_frames / self.video_player.fps) if self.video_player.fps > 0 else "--:--"
                self.left_panel.set_time_display(current_time_str, total_time_str)
                
        except Exception as e:
            print(f"更新进度失败: {e}")
    
    def _on_progress_changed(self, value):
        """用户拖动进度条"""
        if self.current_mode == 'video' and hasattr(self.video_player, 'cap') and self.video_player.cap:
            try:
                import cv2
                # 跳转到指定位置
                total_frames = self.video_player.total_frames
                if total_frames > 0:
                    # value是0-1000，转换为帧号
                    target_frame = int((value / 1000.0) * total_frames)
                    # 修改为使用播放器的seek_frame方法
                    self.video_player.seek_frame(target_frame)
                    
                    print(f"跳转到进度: {value}/1000, 帧号: {target_frame}/{total_frames}")
            except Exception as e:
                print(f"跳转进度失败: {e}")
    
    def _on_play_pause_clicked(self):
        """播放/暂停按钮点击"""
        try:
            if self.current_mode == 'video':
                if self.video_player.playing:
                    self.video_player.pause()
                    self.left_panel.set_play_state(False)
                else:
                    self.video_player.resume()
                    self.left_panel.set_play_state(True)
            elif self.current_mode == 'camera':
                if self.video_player.playing:
                    self.video_player.pause()
                    self.left_panel.set_play_state(False)
                else:
                    self.video_player.resume()
                    self.left_panel.set_play_state(True)
        except Exception as e:
            print(f"播放/暂停失败: {e}")
    
    # ============================================================================
    # 文件菜单处理方法
    # ============================================================================
    
    def _on_file_init(self):
        """初始化"""
        reply = QMessageBox.question(
            self.ui, "确认初始化",
            "是否要初始化所有设置？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._stop_all()
            self._init_ui_state()
            QMessageBox.information(self.ui, "初始化完成", "所有设置已重置")
    
    def _on_file_exit(self):
        """退出"""
        reply = QMessageBox.question(
            self.ui, "确认退出",
            "是否要退出YOLO检测系统？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._stop_all()
            self.ui.close()
    
    # ============================================================================
    # 帮助菜单处理方法
    # ============================================================================
    
    def _on_help_about(self):
        """显示关于对话框"""
        about_text = f"""
        <h3>YOLO多功能检测系统</h3>
        
        <b>版本:</b> 1.0.0<br>
        <b>作者:</b> Sephoration<br><br>
        
        <b>功能特点:</b><br>
        • 目标检测与跟踪<br>
        • 关键点/姿态检测<br>
        • 图像分类<br>
        • 支持图片、视频、摄像头<br>
        • 实时统计与可视化<br><br>
        
        <b>技术支持:</b><br>
        • PySide6 (Qt for Python)<br>
        • Ultralytics YOLO<br>
        • OpenCV<br><br>
        
        <b>© 2024 版权所有</b>
        """
        
        QMessageBox.about(self.ui, "关于", about_text)
    
    def _on_help_manual(self):
        """显示使用说明"""
        manual_text = """
        <h3>YOLO多功能检测系统 - 使用说明</h3>
        
        <b>1. 加载模型</b><br>
        • 点击"打开模型"按钮选择YOLO模型文件 (.pt)<br>
        • 选择对应的模块类型：分析器(目标检测)、分类器、关键点检测<br><br>
        
        <b>2. 打开媒体文件</b><br>
        • <b>图片</b>: 点击"打开图片"，选择图片文件<br>
        • <b>视频</b>: 点击"打开视频"，选择视频文件<br>
        • <b>摄像头</b>: 点击"打开摄像头"，使用默认摄像头<br><br>
        
        <b>3. 参数设置</b><br>
        • <b>IOU阈值</b>: 控制检测框重叠度 (0.0-1.0)<br>
        • <b>置信度</b>: 过滤低置信度检测结果 (0.0-1.0)<br>
        • <b>延迟(ms)</b>: 控制处理间隔，影响实时性<br>
        • <b>线宽</b>: 调整检测框和关键点的绘制线宽<br><br>
        
        <b>4. 开始检测</b><br>
        • 点击"开始"按钮开始推理处理<br>
        • 实时统计面板显示处理结果<br>
        • 点击"停止"按钮结束处理<br><br>
        
        <b>5. 视频控制</b><br>
        • <b>播放/暂停</b>: 控制视频播放<br>
        • <b>进度条</b>: 拖动跳转到指定位置<br>
        • <b>时间显示</b>: 显示当前/总时长<br><br>
        
        <b>6. 其他功能</b><br>
        • <b>保存截图</b>: 保存当前显示画面<br>
        • <b>初始化</b>: 重置所有设置<br>
        • <b>退出</b>: 关闭应用程序<br><br>
        
        <b>提示:</b><br>
        • 确保已安装必要的Python库<br>
        • 使用合适的YOLO模型文件<br>
        • 调整参数以获得最佳检测效果
        """
        
        QMessageBox.information(self.ui, "使用说明", manual_text)
    
    # ============================================================================
    # 主要功能处理方法
    # ============================================================================
    
    def _on_model_load(self):
        """加载模型"""
        try:
            model_filter = "模型文件 (*.pt *.pth *.onnx);;所有文件 (*.*)"
            model_path, _ = QFileDialog.getOpenFileName(
                self.ui, "选择YOLO模型文件",
                "", model_filter
            )
            
            if model_path:
                # 清空之前的模型信息
                self.model_path = None
                self.selected_module_type = None
                self.current_yolo_module = None
                self.model_loaded = False
                
                print(f"开始分析模型: {model_path}")
                
                try:
                    # 使用YOLOAnalyzer分析模型（不加载完整模型）
                    from yolo_analyzer import YOLOAnalyzer
                    
                    # 分析模型信息
                    model_info = YOLOAnalyzer.analyze_model(model_path)
                    
                    # 获取任务类型
                    task_type = model_info.get('task_type', 'detection')
                    
                    # 任务类型到模块类型的映射
                    task_module_map = {
                        'detection': 'analyzer',
                        'classification': 'classifier',
                        'keypoint': 'keypoint',
                        'tracker': 'Tracker',
                        'segmentation': 'analyzer'  # 分割也使用分析器
                    }
                    
                    if task_type not in task_module_map:
                        # 显示选择对话框
                        self._show_model_type_dialog(model_path)
                    else:
                        # 自动确定模块类型
                        self.selected_module_type = task_module_map[task_type]
                        self.model_path = model_path
                        
                        # 获取显示信息
                        display_info = YOLOAnalyzer.get_model_info_for_display(model_info)
                        
                        # 更新UI显示模型信息（但不加载模型）
                        self.right_panel.update_model_info(
                            model_path=model_path,
                            task_type=display_info['task_type'],
                            input_size=display_info['input_size'],
                            class_count=display_info['class_count']
                        )
                        
                        # 显示成功信息
                        QMessageBox.information(
                            self.ui, "模型分析成功",
                            f"✅ 已自动识别模型类型\n\n"
                            f"📦 模型名称: {display_info['model_name']}\n"
                            f"🎯 任务类型: {display_info['task_type']}\n"
                            f"📏 输入尺寸: {display_info['input_size']}\n"
                            f"🔢 类别数量: {display_info['class_count']}\n"
                            f"💾 文件大小: {display_info['file_size']}\n\n"
                            f"模型将在点击'开始'时正式加载。"
                        )
                        
                        print(f"模型分析完成，类型: {self.selected_module_type}")
                    
                except Exception as e:
                    print(f"模型分析失败: {e}")
                    # 分析失败，显示选择对话框
                    self._show_model_type_dialog(model_path)
                    
        except Exception as e:
            self._show_error("选择模型失败", str(e))
    
    def _show_model_type_dialog(self, model_path):
        """显示模型类型选择对话框（当自动识别失败时）"""
        try:
            dialog = QDialog(self.ui)
            dialog.setWindowTitle("选择模型类型")
            dialog.setFixedSize(300, 220)
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)
            
            model_name = os.path.basename(model_path)
            info_label = QLabel(f"已选择模型:\n{model_name}")
            info_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(info_label)
            
            tip_label = QLabel("请选择处理模块:")
            tip_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(tip_label)
            
            btn_analyzer = QPushButton("分析器 (目标检测)")
            btn_classifier = QPushButton("分类器 (图像分类)")
            btn_keypoint = QPushButton("关键点检测 (姿态)")
            btn_tracker = QPushButton("目标跟踪")
            
            button_style = """
                QPushButton {
                    background-color: #f0f0f0;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    padding: 10px;
                    font-weight: normal;
                    min-height: 40px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                    border-color: #aaaaaa;
                }
            """
            
            for btn in [btn_analyzer, btn_classifier, btn_keypoint, btn_tracker]:
                btn.setStyleSheet(button_style)
            
            btn_analyzer.clicked.connect(lambda: self._select_module_type('analyzer', model_path, dialog))
            btn_classifier.clicked.connect(lambda: self._select_module_type('classifier', model_path, dialog))
            btn_keypoint.clicked.connect(lambda: self._select_module_type('keypoint', model_path, dialog))
            btn_tracker.clicked.connect(lambda: self._select_module_type('Tracker', model_path, dialog))
            
            layout.addWidget(btn_analyzer)
            layout.addWidget(btn_classifier)
            layout.addWidget(btn_keypoint)
            layout.addWidget(btn_tracker)
            
            dialog.exec()
            
        except Exception as e:
            self._show_error("选择模型类型失败", str(e))
    
    def _select_module_type(self, module_type: str, model_path: str, dialog):
        """选择模块类型"""
        try:
            self.selected_module_type = module_type
            self.model_path = model_path
            
            module_display_names = {
                'analyzer': '目标检测',
                'classifier': '图像分类',
                'keypoint': '关键点检测',
                'Tracker': '目标跟踪'
            }
            
            display_name = module_display_names.get(module_type, module_type)
            
            # 更新UI显示模型信息（但不加载模型）
            self.right_panel.update_model_info(
                model_path=model_path,
                task_type=display_name,
                input_size="640x640",  # 默认尺寸
                class_count="待检测"
            )
            
            # 显示成功信息
            QMessageBox.information(
                self.ui, "模型选择成功",
                f"✅ 已选择{display_name}模块\n\n"
                f"📦 模型: {os.path.basename(model_path)}\n"
                f"🎯 任务: {display_name}\n\n"
                f"模型将在点击'开始'时正式加载。"
            )
            
            print(f"已选择{display_name}模块，模型将在点击'开始'时加载")
            
            dialog.close()
            
        except Exception as e:
            self._show_error("选择模块失败", str(e))
    
    def _on_image_open(self):
        """打开图片"""
        try:
            self._stop_all()
            
            image_filter = "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*.*)"
            image_path, _ = QFileDialog.getOpenFileName(
                self.ui, "选择图片文件",
                "", image_filter
            )
            
            if image_path:
                self.current_file = image_path
                self.current_mode = 'image'
                
                self.left_panel.update_info(os.path.basename(image_path), 'image')
                
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    self.left_panel.set_display_image(pixmap)
                    print(f"已加载图片: {os.path.basename(image_path)}")
                else:
                    QMessageBox.warning(self.ui, "警告", "无法加载图片文件")
                
        except Exception as e:
            self._show_error("打开图片失败", str(e))
    
    def _on_video_open(self):
        """打开视频"""
        try:
            self._stop_all()

            video_filter = "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*.*)"
            video_path, _ = QFileDialog.getOpenFileName(
                self.ui, "选择视频文件",
                "", video_filter
            )

            if video_path:
                self.current_file = video_path
                self.current_mode = 'video'
                self.is_playing = True

                self.left_panel.update_info(os.path.basename(video_path), 'video')

                # 启动极简播放器
                self.video_player.play_video(video_path)

            print(f"开始播放视频: {os.path.basename(video_path)}")
                
        except Exception:
            self._show_error("打开视频失败", "无法打开视频文件")

    def _on_camera_open(self):
        """打开摄像头"""
        try:
            self._stop_all()

            camera_id = 0  # 默认摄像头

            self.current_file = f"摄像头 {camera_id}"
            self.current_mode = 'camera'
            self.is_playing = True

            self.left_panel.update_info(f"摄像头 {camera_id}", 'camera')

            # 启动极简播放器
            self.video_player.play_camera(camera_id)

            print(f"开始摄像头实时显示")
                
        except Exception:
            self._show_error("打开摄像头失败", "无法打开摄像头")
    
    # ============================================================================
    # 控制按钮处理方法
    # ============================================================================
    
    def _on_start_inference(self):
        """开始推理"""
        try:
            # 检查必要条件
            if not self.current_file:
                QMessageBox.warning(self.ui, "警告", "请先选择媒体文件！")
                return
            
            if not self.model_path or not self.selected_module_type:
                QMessageBox.warning(self.ui, "警告", "请先选择模型和模块类型！")
                return
            
            # 加载模型（此时才真正加载）
            if not self._load_yolo_module():
                return
            
            # 检查当前模式
            if self.current_mode == 'image':
                self._process_image()
            elif self.current_mode in ['video', 'camera']:
                self._process_video_camera()
            else:
                QMessageBox.warning(self.ui, "警告", "请先选择媒体文件！")
                
        except Exception as e:
            self._show_error("开始处理失败", str(e))
    
    def _process_image(self):
        """处理图片"""
        try:
            if not self._load_yolo_module():
                return
            
            print(f"开始处理图片: {self.current_file}")
            
            # 加载图片
            import cv2
            import numpy as np
            from PySide6.QtGui import QImage, QPixmap
            
            image = cv2.imread(self.current_file)
            if image is None:
                QMessageBox.warning(self.ui, "警告", "无法读取图片文件")
                return
            
            # 调用YOLO模块处理图片（返回字典，不是元组）
            result_dict = self.current_yolo_module.process_frame(image)
            
            # 提取处理后的图像和统计信息
            if isinstance(result_dict, dict):
                processed_image = result_dict.get('image', image)
                stats_data = result_dict.get('stats', {})
                
                # 检查是否有具体的分类结果
                class_name = result_dict.get('class_name', '未知')
                confidence = result_dict.get('confidence', 0.0)
                
                # 如果有分类信息，在统计信息中添加
                if class_name != '未知':
                    stats_data['detection_count'] = 1 if confidence > 0 else 0
                    stats_data['avg_confidence'] = confidence
                    stats_data['class_name'] = class_name
                    
                    print(f"分类结果: {class_name} ({confidence:.2%})")
            else:
                # 如果不是字典，使用原始图像
                processed_image = image
                stats_data = {}
                print("警告: YOLO模块返回的不是字典格式")
            
            # 显示处理后的图像
            if isinstance(processed_image, np.ndarray):
                # 确保图像是RGB格式
                if len(processed_image.shape) == 2:  # 灰度图
                    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
                elif processed_image.shape[2] == 4:  # RGBA
                    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2RGB)
                else:  # BGR
                    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                
                height, width, channel = processed_rgb.shape
                bytes_per_line = 3 * width
                
                q_img = QImage(processed_rgb.data, width, height, 
                              bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.left_panel.set_display_image(pixmap)
            else:
                print(f"警告: 处理后的图像不是numpy数组，类型: {type(processed_image)}")
            
            # 更新统计信息
            self.right_panel.update_statistics(
                detection_count=stats_data.get('detection_count', 0),
                confidence=stats_data.get('avg_confidence', 0.0),
                inference_time=stats_data.get('inference_time', 0),
                fps=stats_data.get('fps', 0.0)
            )
            
            # 显示分类结果
            if 'class_name' in stats_data and stats_data['class_name'] != '未知':
                self.right_panel.update_detailed_stats(
                    total_processed=1,
                    total_detections=1 if stats_data.get('detection_count', 0) > 0 else 0,
                    avg_inference_time=0,
                    class_distribution=f"{stats_data['class_name']}: {stats_data.get('avg_confidence', 0.0):.2%}"
                )
            
            # 更新UI状态
            self.right_panel.set_control_state(True)
            self.is_processing = True
            
            print(f"图片处理完成: {self.current_file}")
            
        except Exception as e:
            self._show_error("图片处理失败", str(e))
    
    def _process_video_camera(self):
        """处理视频/摄像头"""
        try:
            if not self._load_yolo_module():
                return
            
            # 设置YOLO模块到抓取器
            self.frame_grabber.set_yolo_module(self.current_yolo_module)
            
            # 获取抓取间隔参数
            delay_ms = self.right_panel.get_parameters().get('delay_ms', 10)
            grab_interval = max(1, delay_ms // 10)  # 根据延迟计算间隔
            
            # 开始抓取帧
            self.frame_grabber.start_grabbing(grab_interval)
            
            # 更新UI状态
            self.right_panel.set_control_state(True)
            self.is_processing = True
            
            print(f"开始处理{self.current_mode}: {self.current_file}")
            print(f"抓取间隔: 每{grab_interval}帧抓取一次")
            
        except Exception as e:
            self._show_error("开始处理失败", str(e))
    
    def _on_stop_inference(self):
        """停止推理"""
        self._stop_processing()
    
    def _on_save_screenshot(self):
        """保存截图"""
        try:
            pixmap = self.left_panel.display_label.pixmap()
            if pixmap and not pixmap.isNull():
                file_filter = "PNG图片 (*.png);;JPEG图片 (*.jpg *.jpeg);;所有文件 (*.*)"
                
                if self.current_file:
                    base_name = os.path.splitext(os.path.basename(self.current_file))[0]
                else:
                    base_name = "screenshot"
                
                default_name = f"{base_name}.png"
                
                save_path, _ = QFileDialog.getSaveFileName(
                    self.ui, "保存截图",
                    default_name,
                    file_filter
                )
                
                if save_path:
                    if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        save_path += '.png'
                    
                    success = pixmap.save(save_path)
                    if success:
                        QMessageBox.information(self.ui, "保存成功", f"截图已保存到:\n{save_path}")
                        print(f"截图保存到: {save_path}")
                    else:
                        QMessageBox.warning(self.ui, "保存失败", "无法保存截图")
            else:
                QMessageBox.warning(self.ui, "警告", "没有可保存的图像")
                
        except Exception as e:
            self._show_error("保存截图失败", str(e))
    
    # ============================================================================
    # YOLO模块加载方法
    # ============================================================================
    
    def _load_yolo_module(self) -> bool:
        """加载YOLO模块（在点击"开始"时调用）"""
        try:
            if not self.model_path or not self.selected_module_type:
                QMessageBox.warning(self.ui, "警告", "请先选择模型和模块类型！")
                return False
            
            # 模块映射
            module_map = {
                'analyzer': 'yolo_analyzer',
                'classifier': 'yolo_classifier',
                'keypoint': 'yolo_keypoint',
                'Tracker': 'yolo_Tracker',
            }
            
            if self.selected_module_type not in module_map:
                self._show_error("加载失败", f"未知的模块类型: {self.selected_module_type}")
                return False
            
            module_file = module_map[self.selected_module_type]
            
            # 动态导入模块
            try:
                yolo_module = importlib.import_module(module_file)
                
                # 类名规则: YOLO{模块名}
                class_name = f"YOLO{self.selected_module_type.capitalize()}"
                if hasattr(yolo_module, class_name):
                    yolo_class = getattr(yolo_module, class_name)
                    
                    # 获取参数
                    params = self.right_panel.get_parameters()
                    
                    print(f"正在正式加载YOLO模型: {self.model_path}")
                    print(f"模块类型: {self.selected_module_type}")
                    print(f"参数: IOU={params['iou_threshold']}, 置信度={params['confidence_threshold']}")
                    
                    # 创建实例（此时才真正加载模型）
                    self.current_yolo_module = yolo_class(
                        model_path=self.model_path,
                        iou_threshold=params['iou_threshold'],
                        confidence_threshold=params['confidence_threshold'],
                        device='cpu'  # 默认使用CPU
                    )
                    
                    self.model_loaded = True
                    
                    # 获取详细的模型信息
                    model_info = {}
                    if hasattr(self.current_yolo_module, 'model_info'):
                        model_info = self.current_yolo_module.model_info
                    elif hasattr(self.current_yolo_module, 'model'):
                        # 尝试从YOLO模型中提取信息
                        try:
                            from ultralytics import YOLO
                            yolo_model = self.current_yolo_module.model
                            model_info['input_size'] = (640, 640)  # 默认
                            model_info['num_classes'] = len(yolo_model.names) if hasattr(yolo_model, 'names') else '未知'
                        except:
                            pass
                    
                    # 获取输入尺寸
                    input_size = model_info.get('input_size', 640)
                    if isinstance(input_size, (list, tuple)):
                        input_size_str = f"{input_size[0]}x{input_size[1]}"
                    else:
                        input_size_str = f"{input_size}x{input_size}"
                    
                    # 获取类别数量
                    class_count = model_info.get('num_classes', '未知')
                    
                    # 更新UI显示详细模型信息
                    module_display_names = {
                        'analyzer': '目标检测',
                        'classifier': '图像分类',
                        'keypoint': '关键点检测',
                        'Tracker': '目标跟踪'
                    }
                    
                    display_name = module_display_names.get(self.selected_module_type, self.selected_module_type)
                    self.right_panel.update_model_info(
                        model_path=self.model_path,
                        task_type=display_name,
                        input_size=input_size_str,
                        class_count=str(class_count)
                    )
                    
                    print(f"✅ YOLO模块加载成功: {self.selected_module_type}")
                    print(f"   - 输入尺寸: {input_size_str}")
                    print(f"   - 类别数量: {class_count}")
                    return True
                else:
                    raise AttributeError(f"模块中没有找到类 {class_name}")
                    
            except ImportError as e:
                self._show_error("导入失败", f"无法导入模块 {module_file}:\n{str(e)}\n请确保{module_file}.py文件存在")
                return False
            except Exception as e:
                self._show_error("加载YOLO模块失败", str(e))
                self.model_loaded = False
                return False
                
        except Exception as e:
            self._show_error("加载YOLO模块失败", str(e))
            self.model_loaded = False
            return False
    
    # ============================================================================
    # 辅助方法
    # ============================================================================
    
    def _stop_all(self):
        """停止所有处理"""
        self._stop_processing()
        self.video_player.stop()
        self.is_playing = False
        self.left_panel.set_controls_enabled(False)
    
    def _stop_processing(self):
        """停止处理"""
        self.frame_grabber.stop_grabbing()
        self.is_processing = False
        self.right_panel.set_control_state(False)
        print("处理已停止")
    
    def _format_time(self, seconds):
        """格式化时间为 MM:SS 格式"""
        try:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        except:
            return "--:--"
    
    def _show_error(self, title: str, message: str):
        """显示错误"""
        QMessageBox.critical(
            self.ui, title,
            f"{message}\n\n详细信息请查看控制台输出。"
        )
        print(f"错误 [{title}]: {message}")
        traceback.print_exc()