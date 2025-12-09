"""
YOLO 关键点检测
直接调用模型对象，禁止使用.predict()方法
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List
from yolo_analyzer import YOLOAnalyzer


class YOLOKeypoint(YOLOAnalyzer):
    """YOLO关键点检测器 - 专门用于姿态估计"""
    
    def __init__(self, model_path: str = None):
        """初始化关键点检测器"""
        super().__init__(model_path, model_type='pose')
        self.keypoint_radius = 3
        self.keypoint_thickness = -1  # 填充
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 脸部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 躯干
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # 四肢
        ]
        self.skeleton_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
        ]
    
    def inference(self, input_data: Union[str, np.ndarray], 
                  conf: float = None, iou: float = None, **kwargs) -> Any:
        """
        执行关键点检测推理 - ✅ 老师的方式：直接调用模型对象
        
        Args:
            input_data: 输入图像
            conf: 置信度阈值
            iou: IOU阈值
            
        Returns:
            Any: 推理结果
        """
        # 使用传入参数或默认值
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou
        
        # ✅ 老师的方式：直接调用模型对象，不使用 .predict()
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
        后处理关键点检测结果
        
        Args:
            results: 推理结果
            original_img: 原始图像
            
        Returns:
            Dict: 处理后的结果
        """
        if len(results) == 0:
            return {
                'boxes': [],
                'keypoints': [],
                'keypoints_conf': [],
                'num_persons': 0,
                'num_keypoints': 0
            }
        
        # 获取第一个结果
        result = results[0]
        
        if result.boxes is None or result.keypoints is None:
            return {
                'boxes': [],
                'keypoints': [],
                'keypoints_conf': [],
                'num_persons': 0,
                'num_keypoints': 0
            }
        
        # 提取边界框
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
        
        # 提取关键点
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints.xy is not None else []
        keypoints_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else []
        
        # 提取类别名称
        class_names = []
        for cls_id in class_ids:
            if hasattr(result, 'names') and cls_id < len(result.names):
                class_names.append(result.names[cls_id])
            else:
                class_names.append(f"person_{cls_id}")
        
        # 构建详细的关键点信息
        detailed_keypoints = []
        for i in range(len(keypoints)):
            person_kps = []
            for j in range(len(keypoints[i])):
                kp = keypoints[i][j]
                conf = keypoints_conf[i][j] if j < len(keypoints_conf[i]) else 0.0
                person_kps.append({
                    'id': j,
                    'x': float(kp[0]),
                    'y': float(kp[1]),
                    'confidence': float(conf),
                    'visible': conf > 0.1
                })
            detailed_keypoints.append(person_kps)
        
        return {
            'boxes': boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
            'confidences': confidences.tolist() if isinstance(confidences, np.ndarray) else confidences,
            'class_ids': class_ids.tolist() if isinstance(class_ids, np.ndarray) else class_ids,
            'class_names': class_names,
            'keypoints': detailed_keypoints,
            'keypoints_array': keypoints,
            'keypoints_conf': keypoints_conf,
            'num_persons': len(boxes),
            'num_keypoints': keypoints.shape[1] if len(keypoints) > 0 else 0,
            'model_has_names': hasattr(result, 'names')
        }
    
    def estimate_pose(self, image_path: str, conf: float = None) -> Dict[str, Any]:
        """
        估计单张图像的姿态
        
        Args:
            image_path: 图像路径
            conf: 置信度阈值
            
        Returns:
            Dict: 姿态估计结果
        """
        results = self.process(image_path, conf=conf)
        
        if results['num_persons'] > 0:
            print(f"检测到 {results['num_persons']} 个人体姿态")
            for i in range(results['num_persons']):
                visible_kps = sum(1 for kp in results['keypoints'][i] if kp['visible'])
                print(f"  第{i+1}人: {visible_kps}/{results['num_keypoints']} 个关键点可见")
        else:
            print("未检测到人体姿态")
        
        return results
    
    def visualize_pose(self, image: np.ndarray, results: Dict[str, Any], 
                      draw_skeleton: bool = True, draw_keypoints: bool = True) -> np.ndarray:
        """
        可视化姿态估计结果
        
        Args:
            image: 原始图像
            results: 姿态估计结果
            draw_skeleton: 是否绘制骨架连接
            draw_keypoints: 是否绘制关键点
            
        Returns:
            np.ndarray: 可视化图像
        """
        vis_img = image.copy()
        
        if results['num_persons'] == 0:
            cv2.putText(vis_img, "No Persons Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_img
        
        # 为每个人绘制
        for person_idx in range(results['num_persons']):
            # 绘制边界框
            if len(results['boxes']) > person_idx:
                box = results['boxes'][person_idx]
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 显示置信度
                if len(results['confidences']) > person_idx:
                    conf = results['confidences'][person_idx]
                    label = f"Person {person_idx+1}: {conf:.2f}"
                    cv2.putText(vis_img, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 获取关键点
            if len(results['keypoints']) > person_idx:
                keypoints = results['keypoints'][person_idx]
                
                if draw_skeleton:
                    # 绘制骨架连接
                    for connection in self.skeleton_connections:
                        start_idx, end_idx = connection
                        if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                            start_kp = keypoints[start_idx]
                            end_kp = keypoints[end_idx]
                            
                            if start_kp['visible'] and end_kp['visible']:
                                color = self.skeleton_colors[connection[0] % len(self.skeleton_colors)]
                                cv2.line(vis_img, 
                                        (int(start_kp['x']), int(start_kp['y'])),
                                        (int(end_kp['x']), int(end_kp['y'])),
                                        color, 2)
                
                if draw_keypoints:
                    # 绘制关键点
                    for kp in keypoints:
                        if kp['visible']:
                            # 根据置信度设置颜色
                            color_intensity = int(255 * kp['confidence'])
                            color = (0, color_intensity, 255 - color_intensity)
                            
                            cv2.circle(vis_img, 
                                      (int(kp['x']), int(kp['y'])),
                                      self.keypoint_radius, color, self.keypoint_thickness)
                            
                            # 可选：显示关键点ID
                            # cv2.putText(vis_img, str(kp['id']), 
                            #            (int(kp['x']), int(kp['y'] - 5)),
                            #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # 添加统计信息
        stats_text = f"Persons: {results['num_persons']} | Keypoints: {results['num_keypoints']}"
        cv2.putText(vis_img, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if 'inference_time' in results:
            time_text = f"Time: {results['inference_time']:.3f}s"
            cv2.putText(vis_img, time_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_img
    
    def analyze_movement(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析两帧之间的运动
        
        Args:
            results1: 第一帧结果
            results2: 第二帧结果
            
        Returns:
            Dict: 运动分析结果
        """
        # 简化的运动分析（可根据需要扩展）
        movement_info = {
            'person_count_change': results2['num_persons'] - results1['num_persons'],
            'movement_detected': False
        }
        
        # 检查关键点位置变化
        if results1['num_persons'] > 0 and results2['num_persons'] > 0:
            # 这里可以实现更复杂的运动分析
            pass
        
        return movement_info