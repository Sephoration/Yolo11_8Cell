# 测试路径管理模块
import os
import sys

# 确保可以导入path_manager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入路径管理模块
from path_manager import PathManager, get_path, get_root_dir

# 测试函数
def test_path_manager():
    print("===== 测试路径管理模块 =====")
    
    # 测试get_root_dir函数
    root_dir = get_root_dir()
    print(f"项目根目录: {root_dir}")
    
    # 测试get_path函数
    paths_to_test = [
        ("datasets_small", "测试small_datasets路径"),
        ("datasets_full", "测试full_datasets路径"),
        ("models_small", "测试models_small路径"),
        ("datasets_small", "eosinophil", "eosinophil.yaml", "测试特定细胞类型配置文件路径")
    ]
    
    for path_args, description in paths_to_test:
        path = get_path(*path_args)
        print(f"{description}: {path}")
        # 检查路径是否存在
        exists = os.path.exists(path)
        print(f"  路径是否存在: {'是' if exists else '否'}")
    
    # 测试PathManager单例
    print("\n===== 测试PathManager单例 =====")
    manager1 = PathManager.get_instance()
    manager2 = PathManager.get_instance()
    print(f"单例测试: {'通过' if manager1 is manager2 else '失败'}")
    
    # 打印所有预定义路径
    print("\n===== 预定义路径 =====")
    for key, path in manager1.get_all_paths().items():
        print(f"{key}: {path}")

if __name__ == "__main__":
    test_path_manager()
