```python
"""
Semantic-SAM 適配器
重新命名自 ssam_progressive_adapter.py，更簡潔的名稱

Author: Rich Kung
Updated: 2025-01-XX
"""

# 更新導入
from .semantic_refinement import progressive_refinement_masks

def generate_with_progressive(frames_dir, selected_frames, sam_ckpt_path, levels, ...):
    """包裝 progressive_refinement_masks，逐幀產生各層候選並補上 gap-fill 遮罩"""
    # ...existing code...
```