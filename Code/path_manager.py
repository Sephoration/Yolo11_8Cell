# 路径管理模块 - 用于统一处理项目中的所有路径
import os
import yaml
from typing import Dict, Optional

class PathManager:
    """
    路径管理类，负责统一处理项目中的所有路径
    支持根据项目位置动态计算路径，避免硬编码绝对路径
    """
    
    def __init__(self, root_dir: Optional[str] = None):
        """
        初始化路径管理器
        
        Args:
            root_dir: 项目根目录，如果为None，则自动从当前文件位置推断
        """
        if root_dir is None:
            # 从当前文件位置推断项目根目录
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            self.root_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
        else:
            self.root_dir = os.path.abspath(root_dir)
        
        # 初始化所有路径
        self._init_paths()
    
    def _init_paths(self):
        """
        初始化所有项目路径
        """
        # 核心目录路径
        self.code_dir = os.path.join(self.root_dir, "Code")
        self.pyside6_dir = os.path.join(self.root_dir, "PySide6")
        self.yolo_gui_dir = os.path.join(self.pyside6_dir, "yolo_gui")
        
        # 数据集目录
        self.datasets_full_dir = os.path.join(self.root_dir, "datasets_full")
        self.datasets_small_dir = os.path.join(self.root_dir, "datasets_small")
        
        # 模型目录
        self.models_full_dir = os.path.join(self.root_dir, "models_full")
        self.models_small_dir = os.path.join(self.root_dir, "models_small")
        
        # 默认模型文件路径
        self.yolo11m_pt = os.path.join(self.code_dir, "yolo11m.pt")
        self.yolo11n_pt = os.path.join(self.code_dir, "yolo11n.pt")
    
    def get_path(self, path_type: str, **kwargs) -> str:
        """
        获取指定类型的路径
        
        Args:
            path_type: 路径类型
            **kwargs: 额外参数
            
        Returns:
            str: 对应的路径
        """
        path_handlers = {
            "root": self.root_dir,
            "code": self.code_dir,
            "datasets_full": self.datasets_full_dir,
            "datasets_small": self.datasets_small_dir,
            "models_full": self.models_full_dir,
            "models_small": self.models_small_dir,
            "yolo_gui": self.yolo_gui_dir,
            "yolo11m": self.yolo11m_pt,
            "yolo11n": self.yolo11n_pt,
            "cell_dataset": self._get_cell_dataset_path,
            "cell_model": self._get_cell_model_path,
            "cell_training": self._get_cell_training_path,
            "cell_yaml": self._get_cell_yaml_path,
        }
        
        handler = path_handlers.get(path_type)
        if handler is None:
            raise ValueError(f"Unknown path type: {path_type}")
        
        if callable(handler):
            return handler(**kwargs)
        return handler
    
    def _get_cell_dataset_path(self, cell_type: str, dataset_type: str = "full", subdir: str = "") -> str:
        """
        获取特定细胞类型的数据集路径
        
        Args:
            cell_type: 细胞类型
            dataset_type: 数据集类型，"full" 或 "small"
            subdir: 子目录，如 "images" 或 "labels"
            
        Returns:
            str: 数据集路径
        """
        base_dir = self.datasets_full_dir if dataset_type == "full" else self.datasets_small_dir
        path = os.path.join(base_dir, cell_type)
        if subdir:
            path = os.path.join(path, subdir)
        return path
    
    def _get_cell_model_path(self, cell_type: str, model_type: str = "small", best: bool = True) -> str:
        """
        获取特定细胞类型的模型路径
        
        Args:
            cell_type: 细胞类型
            model_type: 模型类型，"full" 或 "small"
            best: 是否使用最佳模型
            
        Returns:
            str: 模型路径
        """
        base_dir = self.models_full_dir if model_type == "full" else self.models_small_dir
        model_file = "best.pt" if best else "last.pt"
        return os.path.join(base_dir, f"{cell_type}_train", "weights", model_file)
    
    def _get_cell_training_path(self, cell_type: str, model_type: str = "small") -> str:
        """
        获取特定细胞类型的训练目录路径
        
        Args:
            cell_type: 细胞类型
            model_type: 模型类型，"full" 或 "small"
            
        Returns:
            str: 训练目录路径
        """
        base_dir = self.models_full_dir if model_type == "full" else self.models_small_dir
        return os.path.join(base_dir, f"{cell_type}_train")
    
    def _get_cell_yaml_path(self, cell_type: str, dataset_type: str = "small") -> str:
        """
        获取特定细胞类型的YAML配置文件路径
        
        Args:
            cell_type: 细胞类型
            dataset_type: 数据集类型，"full" 或 "small"
            
        Returns:
            str: YAML文件路径
        """
        base_dir = self.datasets_full_dir if dataset_type == "full" else self.datasets_small_dir
        # 尝试多个可能的YAML文件名
        yaml_paths = [
            os.path.join(base_dir, cell_type, f"{cell_type}.yaml"),
            os.path.join(base_dir, cell_type, "data.yaml")
        ]
        
        for path in yaml_paths:
            if os.path.exists(path):
                return path
        
        # 如果都不存在，返回默认路径
        return yaml_paths[0]
    
    def update_yaml_path(self, yaml_path: str, new_path: Optional[str] = None) -> bool:
        """
        更新YAML文件中的path字段为相对路径或指定的新路径
        
        Args:
            yaml_path: YAML文件路径
            new_path: 新的path值，如果为None，则使用相对于项目根目录的路径
            
        Returns:
            bool: 是否成功更新
        """
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if data is None:
                data = {}
            
            if new_path is None:
                # 计算YAML文件所在目录相对于项目根目录的路径
                yaml_dir = os.path.dirname(yaml_path)
                # 对于数据集YAML，我们使用项目根目录作为path
                data['path'] = self.root_dir
            else:
                data['path'] = new_path
            
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True)
            
            return True
        except Exception as e:
            print(f"Error updating YAML path: {e}")
            return False
    
    def get_all_yaml_files(self, base_dir: Optional[str] = None) -> list:
        """
        获取指定目录下所有的YAML文件
        
        Args:
            base_dir: 基础目录，如果为None则搜索整个项目
            
        Returns:
            list: YAML文件路径列表
        """
        if base_dir is None:
            base_dir = self.root_dir
        
        yaml_files = []
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.yaml'):
                    yaml_files.append(os.path.join(root, file))
        
        return yaml_files
    
    def get_relative_path(self, absolute_path: str) -> str:
        """
        将绝对路径转换为相对于项目根目录的路径
        
        Args:
            absolute_path: 绝对路径
            
        Returns:
            str: 相对路径
        """
        try:
            return os.path.relpath(absolute_path, self.root_dir)
        except ValueError:
            # 如果路径不在同一驱动器，返回原路径
            return absolute_path
    
    def get_absolute_path(self, relative_path: str) -> str:
        """
        将相对于项目根目录的路径转换为绝对路径
        
        Args:
            relative_path: 相对路径
            
        Returns:
            str: 绝对路径
        """
        return os.path.abspath(os.path.join(self.root_dir, relative_path))
    
    def fix_all_yaml_paths(self) -> int:
        """
        修复项目中所有YAML文件的path字段
        
        Returns:
            int: 成功修复的YAML文件数量
        """
        yaml_files = self.get_all_yaml_files()
        fixed_count = 0
        
        for yaml_file in yaml_files:
            if self.update_yaml_path(yaml_file):
                fixed_count += 1
        
        return fixed_count

# 创建全局路径管理器实例
# 这使得模块可以直接作为单例使用
path_manager = PathManager()

# 提供简化的函数接口，方便使用
def get_path(path_type: str, **kwargs) -> str:
    """获取指定类型的路径"""
    return path_manager.get_path(path_type, **kwargs)

def get_root_dir() -> str:
    """获取项目根目录"""
    return path_manager.root_dir

def fix_all_yaml_paths() -> int:
    """修复所有YAML文件的路径"""
    return path_manager.fix_all_yaml_paths()

def get_relative_path(absolute_path: str) -> str:
    """获取相对于项目根目录的路径"""
    return path_manager.get_relative_path(absolute_path)

def get_absolute_path(relative_path: str) -> str:
    """获取绝对路径"""
    return path_manager.get_absolute_path(relative_path)
