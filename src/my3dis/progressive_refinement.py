"""
Progressive Refinement 向後相容層
此檔案僅為保持向後相容性，實際功能已移至專門模組

使用建議：
- 新代碼請直接導入 semantic_refinement 或 semantic_refinement_cli
- 此檔案將在下個版本中被移除

Deprecation Notice:
- progressive_refinement_masks() → use semantic_refinement.progressive_refinement_masks()
- main() → use semantic_refinement_cli.main()
- CLI usage → use semantic_refinement_cli.py directly
"""
import warnings

# 發出棄用警告
warnings.warn(
    "progressive_refinement.py is deprecated. "
    "Use semantic_refinement for algorithms or semantic_refinement_cli for CLI. "
    "This compatibility layer will be removed in the next major version.",
    DeprecationWarning,
    stacklevel=2
)

from .semantic_refinement import (
    DEFAULT_SEMANTIC_SAM_ROOT,
    build_semantic_sam,
    SemanticSamAutomaticMaskGenerator,
    bbox_from_mask,
    console,
    create_experiment_folder,
    create_masked_image,
    get_experiment_timestamp,
    instance_map_to_anns,
    instance_map_to_color_image,
    log_step,
    plot_results,
    prepare_image,
    prepare_image_from_pil,
    progressive_refinement_masks,
    setup_output_directories,
    save_original_image_info,
    timer_decorator,
)
from .semantic_refinement_cli import main
from .common_utils import parse_levels, parse_range


def get_git_commit_hash(default=None):
    """向後相容的 Git commit 取得（已棄用）"""
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "HEAD"])
        return out.decode("utf-8").strip()
    except Exception:
        return default


__all__ = [
    "DEFAULT_SEMANTIC_SAM_ROOT",
    "build_semantic_sam",
    "SemanticSamAutomaticMaskGenerator",
    "bbox_from_mask",
    "console",
    "create_experiment_folder",
    "create_masked_image",
    "get_experiment_timestamp",
    "instance_map_to_anns",
    "instance_map_to_color_image",
    "log_step",
    "plot_results",
    "prepare_image",
    "prepare_image_from_pil",
    "progressive_refinement_masks",
    "setup_output_directories",
    "save_original_image_info",
    "timer_decorator",
    "main",
    "parse_levels",
    "parse_range",
    "get_git_commit_hash",
]

if __name__ == "__main__":
    main()
