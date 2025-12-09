"""
YOLO分析器基类
导入模型时候确定是模型类型，根据模型类型调用不同的程序[分类、目标检测、关键点检测]
要求：
1. 直接调用模型对象，禁止使用.predict()方法
2. 尽量减少显式的参数传递，使用默认值
3. 参数传递要简洁，避免冗长的参数列表
4. 直到开始推理才正式调用对应程序
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Union, List, Optional, Tuple, Dict, Any
import time


class YOLOAnalyzer:
    """YOLO分析器基类 - 遵循老师要求的代码风格"""
    
    def __init__(self, model_path: str = None, model_type: str = 'detect'):
        """
        初始化YOLO分析器
        
        Args:
            model_path: 模型文件路径
            model_type: 模型类型 ('detect', 'pose', 'segment', 'classify')
        """
        self.model = None
        self.model_type = model_type
        self.model_path = model_path
        
        # 推理参数
        self.conf = 0.25
        self.iou = 0.7
        self.img_size = 640
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 如果提供了模型路径，直接加载
        if model_path:
            self.load_model(model_path, model_type)
        
        print(f"YOLO分析器初始化完成 | 类型: {model_type} | 设备: {self.device}")
    
    def load_model(self, model_path: str, model_type: str = None) -> bool:
        """
        加载模型 - 老师的方式：直接创建YOLO对象
        
        Args:
            model_path: 模型文件路径
            model_type: 模型类型，如果为None则自动推断
            
        Returns:
            bool: 是否加载成功
        """
        try:
            print(f"正在加载模型: {model_path}")
            
            # ✅ 老师的方式：直接创建YOLO对象，不使用predict
            self.model = YOLO(model_path)
            
            if model_type:
                self.model_type = model_type
            
            # 移动到指定设备
            self.model.to(self.device)
            
            # 根据模型类型调整默认参数
            self._adjust_params_by_type()
            
            print(f"✅ 模型加载成功: {Path(model_path).name}")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def _adjust_params_by_type(self):
        """根据模型类型调整默认参数"""
        if self.model_type == 'pose':
            self.conf = 0.3
            self.iou = 0.6
        elif self.model_type == 'track':
            self.conf = 0.25
            self.iou = 0.5
        elif self.model_type == 'classify':
            self.conf = 0.25
            self.iou = 0.45
    
    def preprocess_input(self, input_data: Union[str, np.ndarray, List]) -> Tuple[np.ndarray, Any]:
        """
        预处理输入数据
        
        Args:
            input_data: 输入数据，可以是路径、图像数组或路径列表
            
        Returns:
            Tuple: (原始图像, 预处理后的图像)
        """
        if isinstance(input_data, str):
            # 单个图像路径
            img = cv2.imread(input_data)
            if img is None:
                raise ValueError(f"无法加载图像: {input_data}")
            original = img.copy()
            processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return original, processed
            
        elif isinstance(input_data, np.ndarray):
            # numpy数组
            original = input_data.copy()
            if len(input_data.shape) == 3 and input_data.shape[2] == 3:
                processed = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            else:
                processed = input_data
            return original, processed
            
        elif isinstance(input_data, List):
            # 批量处理
            originals = []
            processed_list = []
            for item in input_data:
                if isinstance(item, str):
                    img = cv2.imread(item)
                    if img is not None:
                        originals.append(img.copy())
                        processed_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                elif isinstance(item, np.ndarray):
                    originals.append(item.copy())
                    if len(item.shape) == 3 and item.shape[2] == 3:
                        processed_list.append(cv2.cvtColor(item, cv2.COLOR_BGR2RGB))
                    else:
                        processed_list.append(item)
            
            return originals, processed_list
        
        else:
            raise TypeError(f"不支持的输入类型: {type(input_data)}")
    
    def inference(self, input_data: Union[str, np.ndarray], 
                  conf: float = None, iou: float = None, **kwargs) -> Any:
        """
        执行推理 - 基类方法，子类需要重写
        
        Args:
            input_data: 输入数据
            conf: 置信度阈值
            iou: IOU阈值
            **kwargs: 其他参数
            
        Returns:
            Any: 推理结果
        """
        raise NotImplementedError("子类必须实现 inference 方法")
    
    def postprocess(self, results: Any, original_img: np.ndarray) -> Dict[str, Any]:
        """
        后处理结果 - 基类方法，子类需要重写
        
        Args:
            results: 原始推理结果
            original_img: 原始图像
            
        Returns:
            Dict: 处理后结果
        """
        raise NotImplementedError("子类必须实现 postprocess 方法")
    
    def process(self, input_data: Union[str, np.ndarray], 
                conf: float = None, iou: float = None, **kwargs) -> Dict[str, Any]:
        """
        完整的处理流程：预处理 → 推理 → 后处理
        
        Args:
            input_data: 输入数据
            conf: 置信度阈值
            iou: IOU阈值
            **kwargs: 其他参数
            
        Returns:
            Dict: 处理结果
        """
        if self.model is None:
            raise ValueError("请先加载模型！")
        
        # 1. 预处理
        original_img, processed_img = self.preprocess_input(input_data)
        
        # 2. 推理
        start_time = time.time()
        results = self.inference(processed_img, conf, iou, **kwargs)
        inference_time = time.time() - start_time
        
        # 3. 后处理
        processed_results = self.postprocess(results, original_img)
        processed_results['inference_time'] = inference_time
        processed_results['image_shape'] = original_img.shape
        
        return processed_results
    
    def set_parameters(self, conf: float = None, iou: float = None, 
                      img_size: int = None, device: str = None):
        """设置推理参数"""
        if conf is not None:
            self.conf = conf
        if iou is not None:
            self.iou = iou
        if img_size is not None:
            self.img_size = img_size
        if device is not None:
            self.device = device
            if self.model:
                self.model.to(device)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {"status": "模型未加载"}
        
        info = {
            "model_type": self.model_type,
            "device": self.device,
            "parameters": {
                "conf": self.conf,
                "iou": self.iou,
                "img_size": self.img_size
            }
        }
        
        # 尝试获取模型详细信息
        try:
            if hasattr(self.model, 'names'):
                info['class_names'] = list(self.model.names.values())
                info['num_classes'] = len(self.model.names)
        except:
            pass
        
        return info
    
    def __call__(self, input_data: Union[str, np.ndarray], **kwargs):
        """使对象可调用，更符合老师的简洁风格"""
        return self.process(input_data, **kwargs)