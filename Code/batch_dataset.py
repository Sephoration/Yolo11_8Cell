import os
import shutil
import random
from pathlib import Path
from typing import Dict, List

# ================================================================================
# 配置参数
EXTRACT_COUNT = 200      # 每个类别提取总数
MIN_PER_TYPE = 40        # 每个子类型最少数量
CELL_TYPES = [
    "basophil", "eosinophil", "erythroblast", "ig", 
    "lymphocyte", "monocyte", "neutrophil", "platelet"
]

# ================================================================================
class ImageExtractor:
    def __init__(self):
        self.script_dir = Path(__file__).parent.absolute()
        self.project_root = self.script_dir.parent.absolute()
        self.src_dir = self.project_root / "datasets_full"
        self.dst_dir = self.project_root / "datasets_small"
        
        if not self.src_dir.exists():
            raise FileNotFoundError(f"源目录不存在: {self.src_dir}")
        self.dst_dir.mkdir(exist_ok=True)

    # ============================================================================
    def group_by_prefix(self, img_files: List[Path]) -> Dict[str, List[Path]]:
        """按图片前缀分组（如BNE_001.jpg -> BNE组）"""
        groups = {}
        for img in img_files:
            prefix = img.stem.split('_')[0] if '_' in img.stem else 'default'
            groups.setdefault(prefix, []).append(img)
        return groups

    # ============================================================================
    def allocate_quota(self, groups: Dict[str, List[Path]]) -> Dict[str, int]:
        """分配数量：先各分MIN_PER_TYPE张，剩余按比例分配"""
        print(f"  分配策略: 先各分{MIN_PER_TYPE}张，剩余按比例分配")
        
        allocations = {}
        # 1. 先分最低保障
        for subtype, imgs in groups.items():
            allocations[subtype] = min(MIN_PER_TYPE, len(imgs))
        
        used = sum(allocations.values())
        remaining = EXTRACT_COUNT - used
        
        # 2. 剩余按比例分
        if remaining > 0:
            original_counts = {s: len(imgs) for s, imgs in groups.items()}
            total_original = sum(original_counts.values())
            
            for subtype in groups:
                if allocations[subtype] < len(groups[subtype]):
                    extra = int(remaining * original_counts[subtype] / total_original)
                    capacity = len(groups[subtype]) - allocations[subtype]
                    actual_extra = min(extra, capacity, remaining)
                    
                    if actual_extra > 0:
                        allocations[subtype] += actual_extra
                        remaining -= actual_extra
        
        # 3. 处理取整剩余
        if remaining > 0:
            for subtype in sorted(groups, key=lambda x: len(groups[x]) - allocations[x], reverse=True):
                if remaining <= 0: break
                capacity = len(groups[subtype]) - allocations[subtype]
                if capacity > 0:
                    allocations[subtype] += 1
                    remaining -= 1
        
        # 打印结果
        total = sum(allocations.values())
        print(f"  已分配: {total}/{EXTRACT_COUNT}张")
        for subtype, count in allocations.items():
            print(f"    {subtype}: {count}/{len(groups[subtype])}张")
        
        return allocations

    # ============================================================================
    def extract_cell_type(self, cell_type: str) -> int:
        """提取单个细胞类型的图片"""
        print(f"\n▶ 提取: {cell_type}")
        
        src_path = self.src_dir / cell_type / "images"
        dst_path = self.dst_dir / cell_type / "images"
        
        if not src_path.exists():
            print(f"  ⚠ 跳过: 源目录不存在")
            return 0
        
        # 获取图片
        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            img_files.extend(src_path.glob(ext))
        
        if not img_files:
            print(f"  ⚠ 跳过: 没有图片")
            return 0
        
        print(f"  总图片数: {len(img_files)}张")
        
        # 创建目标目录
        dst_path.mkdir(parents=True, exist_ok=True)
        
        # 分组处理
        groups = self.group_by_prefix(img_files)
        
        if len(groups) == 1:
            # 单类型，直接提取
            count = min(EXTRACT_COUNT, len(img_files))
            selected = random.sample(img_files, count)
            print(f"  单一类型，提取{count}张")
        else:
            # 多类型，分配数量
            print(f"  发现{len(groups)}种子类型: {list(groups.keys())}")
            allocations = self.allocate_quota(groups)
            
            selected = []
            for subtype, count in allocations.items():
                selected.extend(random.sample(groups[subtype], count))
        
        # 复制图片
        for img in selected:
            shutil.copy2(img, dst_path / img.name)
        
        print(f"  ✅ 完成: 复制{len(selected)}张图片")
        return len(selected)

    # ============================================================================
    def run(self):
        """运行提取流程"""
        print("=" * 60)
        print("图片提取工具")
        print("=" * 60)
        print(f"每类提取: {EXTRACT_COUNT}张")
        print(f"每子类型最少: {MIN_PER_TYPE}张")
        print(f"处理 {len(CELL_TYPES)} 个类型")
        print("=" * 60)
        
        total = 0
        for cell_type in CELL_TYPES:
            total += self.extract_cell_type(cell_type)
        
        print("\n" + "=" * 60)
        print(f"✅ 完成! 共提取 {total} 张图片")
        print("=" * 60)

# ================================================================================
def main():
    try:
        extractor = ImageExtractor()
        extractor.run()
    except Exception as e:
        print(f"❌ 错误: {e}")

# ================================================================================
if __name__ == "__main__":
    main()