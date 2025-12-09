"""
YOLO 目标检测
直接调用模型对象，禁止使用.predict()方法
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List, Optional
from yolo_analyzer import YOLOAnalyzer


class YOLOTracker(YOLOAnalyzer):
    """YOLO目标跟踪器 - 专门用于目标检测和跟踪"""
    
    def __init__(self, model_path: str = None, tracker_config: str = "bytetrack.yaml"):
        """初始化跟踪器"""
        super().__init__(model_path, model_type='track')
        self.tracker_config = tracker_config
        self.persist_tracks = True
        self.track_history = {}  # 跟踪历史记录
        self.max_history_length = 50  # 最大历史长度
        self.track_colors = {}  # 为每个track_id分配颜色
    
    def inference(self, input_data: Union[str, np.ndarray], 
                  conf: float = None, iou: float = None, 
                  mode: str = 'track', **kwargs) -> Any:
        """
        执行推理 - ✅ 老师的方式：直接调用模型对象
        
        Args:
            input_data: 输入图像
            conf: 置信度阈值
            iou: IOU阈值
            mode: 模式 ('track' 或 'detect')
            
        Returns:
            Any: 推理结果
        """
        # 使用传入参数或默认值
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou
        
        if mode == 'track':
            # ✅ 老师的方式：使用 .track() 方法进行跟踪
            results = self.model.track(
                input_data,
                conf=conf,
                iou=iou,
                imgsz=self.img_size,
                tracker=self.tracker_config,
                persist=self.persist_tracks,
                verbose=False
            )
        else:
            # ✅ 老师的方式：直接调用模型对象进行检测
            results = self.model(
                input_data,
                conf=conf,
                iou=iou,
                imgsz=self.img_size,
                verbose=False
            )
        
        return results
    
    def postprocess(self, results: Any, original_img: np.ndarray) -> Dict[str, Any]:
        """
        后处理跟踪结果
        
        Args:
            results: 推理结果
            original_img: 原始图像
            
        Returns:
            Dict: 处理后的结果
        """
        if len(results) == 0:
            return {
                'boxes': [],
                'track_ids': [],
                'confidences': [],
                'class_ids': [],
                'class_names': [],
                'num_detections': 0,
                'mode': 'track'
            }
        
        # 获取第一个结果
        result = results[0]
        
        if result.boxes is None:
            return {
                'boxes': [],
                'track_ids': [],
                'confidences': [],
                'class_ids': [],
                'class_names': [],
                'num_detections': 0,
                'mode': 'track'
            }
        
        # 提取边界框
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
        
        # 提取track_id（如果是跟踪模式）
        track_ids = []
        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        
        # 提取类别名称
        class_names = []
        for cls_id in class_ids:
            if hasattr(result, 'names') and cls_id < len(result.names):
                class_names.append(result.names[cls_id])
            else:
                class_names.append(f"object_{cls_id}")
        
        # 更新跟踪历史
        self._update_track_history(track_ids, boxes)
        
        # 为新的track_id分配颜色
        self._assign_track_colors(track_ids)
        
        return {
            'boxes': boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
            'track_ids': track_ids.tolist() if isinstance(track_ids, np.ndarray) else track_ids,
            'confidences': confidences.tolist() if isinstance(confidences, np.ndarray) else confidences,
            'class_ids': class_ids.tolist() if isinstance(class_ids, np.ndarray) else class_ids,
            'class_names': class_names,
            'num_detections': len(boxes),
            'mode': 'track' if len(track_ids) > 0 else 'detect',
            'track_history': self.track_history.copy()
        }
    
    def _update_track_history(self, track_ids: List[int], boxes: List):
        """更新跟踪历史记录"""
        for i, track_id in enumerate(track_ids):
            if i < len(boxes):
                box = boxes[i]
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                center = (center_x, center_y)
                
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append(center)
                
                # 限制历史记录长度
                if len(self.track_history[track_id]) > self.max_history_length:
                    self.track_history[track_id].pop(0)
    
    def _assign_track_colors(self, track_ids: List[int]):
        """为每个track_id分配颜色"""
        colors = [
            (255, 0, 0),    # 蓝色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 黄色
            (128, 0, 0),    # 深蓝
            (0, 128, 0),    # 深绿
            (0, 0, 128),    # 深红
            (128, 128, 0),  # 橄榄色
        ]
        
        for track_id in track_ids:
            if track_id not in self.track_colors:
                color_idx = track_id % len(colors)
                self.track_colors[track_id] = colors[color_idx]
    
    def detect_objects(self, image_path: str, conf: float = None) -> Dict[str, Any]:
        """
        检测图像中的物体
        
        Args:
            image_path: 图像路径
            conf: 置信度阈值
            
        Returns:
            Dict: 检测结果
        """
        results = self.process(image_path, conf=conf, mode='detect')
        
        if results['num_detections'] > 0:
            print(f"检测到 {results['num_detections']} 个物体")
            for i in range(min(5, results['num_detections'])):  # 显示前5个
                class_name = results['class_names'][i]
                confidence = results['confidences'][i]
                print(f"  {class_name}: {confidence:.2f}")
        else:
            print("未检测到任何物体")
        
        return results
    
    def track_objects(self, image_path: str, conf: float = None) -> Dict[str, Any]:
        """
        跟踪图像中的物体
        
        Args:
            image_path: 图像路径
            conf: 置信度阈值
            
        Returns:
            Dict: 跟踪结果
        """
        results = self.process(image_path, conf=conf, mode='track')
        
        if results['num_detections'] > 0:
            print(f"跟踪到 {results['num_detections']} 个物体")
            if results['track_ids']:
                unique_ids = set(results['track_ids'])
                print(f"  唯一ID数量: {len(unique_ids)}")
        else:
            print("未跟踪到任何物体")
        
        return results
    
    def track_video(self, video_path: str, output_path: str = None, 
                   conf: float = None, show: bool = True):
        """
        跟踪视频中的物体
        
        Args:
            video_path: 视频路径
            output_path: 输出视频路径
            conf: 置信度阈值
            show: 是否实时显示
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 准备视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_time = 0
        
        print(f"开始处理视频: {video_path}")
        print(f"视频信息: {width}x{height}, {fps}FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 跟踪当前帧
            start_time = time.time()
            results = self.track_objects(frame, conf=conf)
            frame_time = time.time() - start_time
            total_time += frame_time
            
            # 可视化结果
            vis_frame = self.visualize_tracking(frame, results)
            
            # 显示帧率信息
            fps_text = f"FPS: {1/frame_time:.1f}" if frame_time > 0 else "FPS: N/A"
            cv2.putText(vis_frame, fps_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示处理进度
            progress = f"Frame: {frame_count}"
            cv2.putText(vis_frame, progress, (10, height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示或保存
            if show:
                cv2.imshow('YOLO Tracking', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if writer:
                writer.write(vis_frame)
            
            # 每100帧打印一次进度
            if frame_count % 100 == 0:
                avg_time = total_time / frame_count
                print(f"已处理 {frame_count} 帧，平均每帧 {avg_time:.3f}秒")
        
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        
        print(f"视频处理完成，共处理 {frame_count} 帧")
        if frame_count > 0:
            print(f"平均处理速度: {frame_count/total_time:.1f} FPS")
    
    def visualize_tracking(self, image: np.ndarray, results: Dict[str, Any], 
                          draw_trails: bool = True) -> np.ndarray:
        """
        可视化跟踪结果
        
        Args:
            image: 原始图像
            results: 跟踪结果
            draw_trails: 是否绘制轨迹
            
        Returns:
            np.ndarray: 可视化图像
        """
        vis_img = image.copy()
        
        if results['num_detections'] == 0:
            cv2.putText(vis_img, "No Objects Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_img
        
        # 绘制检测框和轨迹
        for i in range(results['num_detections']):
            # 获取检测信息
            box = results['boxes'][i]
            x1, y1, x2, y2 = map(int, box[:4])
            
            # 获取或分配颜色
            track_id = None
            if i < len(results['track_ids']):
                track_id = results['track_ids'][i]
                color = self.track_colors.get(track_id, (0, 255, 0))
            else:
                color = (0, 255, 0)  # 默认为绿色
            
            # 绘制边界框
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签
            class_name = results['class_names'][i] if i < len(results['class_names']) else f"object"
            confidence = results['confidences'][i] if i < len(results['confidences']) else 0.0
            
            if track_id is not None:
                label = f"ID:{track_id} {class_name} {confidence:.2f}"
            else:
                label = f"{class_name} {confidence:.2f}"
            
            # 绘制标签背景
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(vis_img, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # 绘制标签
            cv2.putText(vis_img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 绘制中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(vis_img, (center_x, center_y), 3, color, -1)
        
        # 绘制轨迹
        if draw_trails and 'track_history' in results:
            for track_id, history in results['track_history'].items():
                if track_id in self.track_colors:
                    color = self.track_colors[track_id]
                    
                    # 绘制轨迹线
                    for j in range(1, len(history)):
                        if history[j-1] is None or history[j] is None:
                            continue
                        
                        thickness = int(np.sqrt(32 / float(j + 1)) * 2)
                        cv2.line(vis_img, history[j-1], history[j], color, thickness)
        
        # 添加统计信息
        stats_text = f"Objects: {results['num_detections']}"
        if results['track_ids']:
            unique_ids = set(results['track_ids'])
            stats_text += f" | Tracks: {len(unique_ids)}"
        
        cv2.putText(vis_img, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if 'inference_time' in results:
            time_text = f"Time: {results['inference_time']:.3f}s"
            cv2.putText(vis_img, time_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_img
    
    def clear_history(self):
        """清除跟踪历史"""
        self.track_history.clear()
        self.track_colors.clear()