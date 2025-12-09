"""
YOLO 分类器
直接调用模型对象，禁止使用.predict()方法
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List
from yolo_analyzer import YOLOAnalyzer


class YOLOClassifier(YOLOAnalyzer):
    """YOLO分类器 - 专门用于图像分类"""
    
    def __init__(self, model_path: str = None):
        """初始化分类器"""
        super().__init__(model_path, model_type='classify')
        self.top_k = 5  # 显示前K个类别
    
    def inference(self, input_data: Union[str, np.ndarray], 
                  conf: float = None, iou: float = None, **kwargs) -> Any:
        """
        执行分类推理 - ✅ 老师的方式：直接调用模型对象
        
        Args:
            input_data: 输入图像
            conf: 置信度阈值（分类任务通常不使用）
            iou: IOU阈值（分类任务通常不使用）
            
        Returns:
            Any: 推理结果
        """
        # 使用传入参数或默认值
        conf = conf if conf is not None else self.conf
        
        # ✅ 老师的方式：直接调用模型对象，不使用 .predict()
        results = self.model(
            input_data,
            conf=conf,
            imgsz=self.img_size,
            verbose=False
        )
        
        return results
    
    def postprocess(self, results: Any, original_img: np.ndarray) -> Dict[str, Any]:
        """
        后处理分类结果
        
        Args:
            results: 推理结果
            original_img: 原始图像
            
        Returns:
            Dict: 处理后的结果
        """
        if len(results) == 0:
            return {
                'predictions': [],
                'top_class': None,
                'top_confidence': 0.0,
                'num_detections': 0
            }
        
        # 获取第一个结果
        result = results[0]
        
        # 提取分类结果
        if hasattr(result, 'probs') and result.probs is not None:
            # 获取概率和类别
            probs = result.probs.data.cpu().numpy()
            top5_indices = np.argsort(probs)[-self.top_k:][::-1]
            top5_probs = probs[top5_indices]
            
            # 获取类别名称
            if hasattr(result, 'names'):
                top5_classes = [result.names[i] for i in top5_indices]
            else:
                top5_classes = [f"class_{i}" for i in top5_indices]
            
            # 构建结果字典
            predictions = []
            for cls, prob in zip(top5_classes, top5_probs):
                predictions.append({
                    'class': cls,
                    'confidence': float(prob),
                    'class_id': int(cls.split('_')[-1]) if cls.startswith('class_') else None
                })
            
            # 最高置信度的类别
            top_idx = top5_indices[0]
            top_class = top5_classes[0]
            top_confidence = float(top5_probs[0])
            
            return {
                'predictions': predictions,
                'top_class': top_class,
                'top_confidence': top_confidence,
                'top_class_id': int(top_idx),
                'num_detections': 1  # 分类任务通常只有一个结果
            }
        
        return {
            'predictions': [],
            'top_class': None,
            'top_confidence': 0.0,
            'num_detections': 0
        }
    
    def classify_image(self, image_path: str, top_k: int = None) -> Dict[str, Any]:
        """
        分类单张图像
        
        Args:
            image_path: 图像路径
            top_k: 返回前K个类别
            
        Returns:
            Dict: 分类结果
        """
        if top_k is not None:
            self.top_k = top_k
        
        # ✅ 简洁调用：直接使用基类的process方法
        results = self.process(image_path)
        
        # 添加一些分类特有的信息
        if results['num_detections'] > 0:
            results['classification_success'] = True
            print(f"分类结果: {results['top_class']} ({results['top_confidence']:.2%})")
        else:
            results['classification_success'] = False
            print("分类失败")
        
        return results
    
    def classify_batch(self, image_paths: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """
        批量分类图像
        
        Args:
            image_paths: 图像路径列表
            top_k: 返回前K个类别
            
        Returns:
            List: 每个图像的分类结果
        """
        if top_k is not None:
            self.top_k = top_k
        
        all_results = []
        for img_path in image_paths:
            try:
                result = self.classify_image(img_path)
                result['image_path'] = img_path
                all_results.append(result)
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                all_results.append({
                    'image_path': img_path,
                    'error': str(e),
                    'classification_success': False
                })
        
        return all_results
    
    def visualize_classification(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        可视化分类结果
        
        Args:
            image: 原始图像
            results: 分类结果
            
        Returns:
            np.ndarray: 可视化图像
        """
        vis_img = image.copy()
        
        if results['num_detections'] == 0:
            cv2.putText(vis_img, "Classification Failed", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_img
        
        # 显示分类结果
        top_class = results['top_class']
        top_confidence = results['top_confidence']
        
        # 主要标签
        main_label = f"{top_class}: {top_confidence:.2%}"
        cv2.putText(vis_img, main_label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 显示前K个结果
        y_offset = 60
        for i, pred in enumerate(results['predictions']):
            label = f"{i+1}. {pred['class']}: {pred['confidence']:.2%}"
            cv2.putText(vis_img, label, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 30
        
        # 添加推理时间
        if 'inference_time' in results:
            time_text = f"Time: {results['inference_time']:.3f}s"
            cv2.putText(vis_img, time_text, (10, y_offset + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_img 