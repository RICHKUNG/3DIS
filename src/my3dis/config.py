
"""
統一配置架構，解決 PROBLEM.md 中提到的配置管理問題

Author: Rich Kung
Updated: 2025-01-XX
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import os
import yaml
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """實驗配置"""
    name: str = "default_experiment"
    scenes: List[str] = field(default_factory=lambda: ["scene_00065_00"])
    dataset_root: str = "./data/multiscan"
    output_root: str = "./outputs"
    tag: Optional[str] = None
    run_dir: Optional[str] = None

    def __post_init__(self):
        # 支援環境變數覆寫
        if "MY3DIS_DATASET_ROOT" in os.environ:
            self.dataset_root = os.environ["MY3DIS_DATASET_ROOT"]
        if "MY3DIS_OUTPUT_ROOT" in os.environ:
            self.output_root = os.environ["MY3DIS_OUTPUT_ROOT"]

@dataclass  
class FrameConfig:
    """影格配置"""
    start: Optional[int] = None
    end: Optional[int] = None
    step: int = 20

@dataclass
class SSAMConfig:
    """Semantic-SAM 配置"""
    levels: List[int] = field(default_factory=lambda: [2, 4, 6])
    min_area: int = 200
    max_masks: int = 2000
    freq: int = 1
    fill_area: Optional[int] = None
    add_gaps: bool = True
    persist_raw: bool = False
    stability_threshold: float = 0.8

@dataclass
class FilterConfig:
    """過濾配置"""
    enabled: bool = True
    min_area: int = 200
    stability_threshold: float = 0.8
    skip_filtering: bool = False

@dataclass
class SAM2Config:
    """SAM2 配置"""
    max_propagate: Optional[int] = None
    prompt_mode: str = "all_mask"  # all_mask, lt_bbox, all_bbox
    iou_threshold: float = 0.6
    downscale_masks: bool = False
    downscale_ratio: float = 0.3
    render_viz: bool = True

@dataclass
class ReportConfig:
    """報告配置"""
    enabled: bool = True
    max_preview_width: int = 640

@dataclass
class StageConfig:
    """階段配置"""
    ssam: Dict[str, Any] = field(default_factory=dict)
    filter: Dict[str, Any] = field(default_factory=dict)
    tracker: Dict[str, Any] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)
    gpu: Optional[int] = None

@dataclass
class PipelineConfig:
    """完整管線配置"""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    frames: FrameConfig = field(default_factory=FrameConfig)
    stages: StageConfig = field(default_factory=StageConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PipelineConfig":
        """從 YAML 檔案載入配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 解析實驗配置
        exp_data = data.get('experiment', {})
        experiment = ExperimentConfig(**exp_data)
        
        # 解析影格配置
        frame_data = data.get('frames', {})
        frames = FrameConfig(**frame_data)
        
        # 解析階段配置
        stage_data = data.get('stages', {})
        stages = StageConfig(**stage_data)
        
        return cls(experiment=experiment, frames=frames, stages=stages)
    
    def get_ssam_config(self) -> SSAMConfig:
        """獲取 SSAM 配置"""
        ssam_data = self.stages.ssam
        # 從 experiment 繼承部分參數
        if 'levels' not in ssam_data and hasattr(self.experiment, 'levels'):
            ssam_data['levels'] = getattr(self.experiment, 'levels', [2, 4, 6])
        return SSAMConfig(**ssam_data)
    
    def get_filter_config(self) -> FilterConfig:
        """獲取過濾配置"""
        return FilterConfig(**self.stages.filter)
    
    def get_sam2_config(self) -> SAM2Config:
        """獲取 SAM2 配置"""
        return SAM2Config(**self.stages.tracker)
    
    def get_report_config(self) -> ReportConfig:
        """獲取報告配置"""
        return ReportConfig(**self.stages.report)
    
    def validate(self) -> None:
        """驗證配置參數"""
        ssam_config = self.get_ssam_config()
        
        if not ssam_config.levels:
            raise ValueError("SSAM levels cannot be empty")
        
        if ssam_config.levels != sorted(ssam_config.levels):
            raise ValueError("SSAM levels must be in ascending order")
        
        if self.frames.step <= 0:
            raise ValueError("Frame step must be positive")
        
        sam2_config = self.get_sam2_config()
        if sam2_config.prompt_mode not in ['all_mask', 'lt_bbox', 'all_bbox']:
            raise ValueError(f"Invalid prompt_mode: {sam2_config.prompt_mode}")
        
        logger.info("Configuration validation passed")

def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """載入並驗證配置"""
    config = PipelineConfig.from_yaml(config_path)
    config.validate()
    return config
