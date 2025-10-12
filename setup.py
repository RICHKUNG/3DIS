"""
My3DIS 套件安裝設定
更新 console entry points 以反映重新命名的模組
"""
from setuptools import setup, find_packages
from pathlib import Path

# ...existing setup code...

setup(
    # ...existing setup parameters...
    
    # 更新主控台入口點以反映新的模組名稱
    entry_points={
        "console_scripts": [
            "my3dis-workflow=my3dis.workflow_runner:main",
            "my3dis-ssam=my3dis.ssam_generator:main", 
            "my3dis-sam2=my3dis.sam2_tracker:main",
            "my3dis-filter=my3dis.candidate_filter:main",
            "my3dis-report=my3dis.report_builder:main",
            "my3dis-refine=my3dis.semantic_refinement_cli:main",
        ],
    },
    
    # ...existing setup parameters...
)
