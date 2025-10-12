#!/usr/bin/env python3
"""
簡單驗證重構後的模組化架構
"""
import warnings
import sys
from pathlib import Path

# 設定 PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("=== 測試重構後的模組化架構 ===")

# 1. 測試新的核心模組導入
print("\n1. 測試核心模組導入...")
try:
    from my3dis.semantic_refinement import progressive_refinement_masks
    print("✅ semantic_refinement 導入成功")
except ImportError as e:
    print(f"❌ semantic_refinement 導入失敗: {e}")

# 2. 測試新的 CLI 模組導入
print("\n2. 測試 CLI 模組導入...")
try:
    from my3dis.semantic_refinement_cli import main, parse_args
    print("✅ semantic_refinement_cli 導入成功")
except ImportError as e:
    print(f"❌ semantic_refinement_cli 導入失敗: {e}")

# 3. 測試統一配置導入
print("\n3. 測試統一配置導入...")
try:
    from my3dis.config import PipelineConfig, load_config
    print("✅ config 模組導入成功")
except ImportError as e:
    print(f"❌ config 模組導入失敗: {e}")

# 4. 測試向後相容層（應該會有棄用警告）
print("\n4. 測試向後相容層...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        from my3dis.progressive_refinement import progressive_refinement_masks as old_func
        from my3dis.progressive_refinement import main as old_main
        
        if len(w) > 0 and "deprecated" in str(w[0].message).lower():
            print("✅ 向後相容層導入成功，棄用警告正常顯示")
            print(f"   警告訊息: {w[0].message}")
        else:
            print("⚠️  向後相容層導入成功，但缺少棄用警告")
            
    except ImportError as e:
        print(f"❌ 向後相容層導入失敗: {e}")

# 5. 測試函數是否為同一個物件（重新導出）
print("\n5. 測試函數重新導出...")
try:
    from my3dis.semantic_refinement import progressive_refinement_masks as new_func
    from my3dis.progressive_refinement import progressive_refinement_masks as old_func
    
    if new_func is old_func:
        print("✅ 函數重新導出正確，新舊函數為同一物件")
    else:
        print("❌ 函數重新導出有問題，新舊函數不同")
        
except ImportError as e:
    print(f"❌ 函數重新導出測試失敗: {e}")

# 6. 測試配置建立
print("\n6. 測試配置建立...")
try:
    from my3dis.config import PipelineConfig
    config = PipelineConfig()
    config.validate()
    print("✅ 預設配置建立與驗證成功")
except Exception as e:
    print(f"❌ 配置建立失敗: {e}")

print("\n=== 測試完成 ===")
