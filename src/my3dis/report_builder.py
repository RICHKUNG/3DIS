
"""
報告建構器
重新命名自 generate_report.py，提供更清晰的模組名稱

Author: Rich Kung
Updated: 2025-01-XX
"""

import os
import json
from pathlib import Path

import click
from PIL import Image

from my3dis.workflow import summary as workflow_summary


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_report(run_dir, report_name, max_preview_width):
    """整合 manifest、summary、視覺化輸出完整報告"""
    run_dir = Path(run_dir)
    output_dir = run_dir / "report"
    output_dir.mkdir(exist_ok=True)

    # 1. 複製 manifest 與 summary
    for fname in ["manifest.json", "workflow_summary.json"]:
        src = run_dir / fname
        dst = output_dir / fname
        if not dst.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # 2. 建立報告索引
    report_index = {"scenes": [], "parameters": {}, "stages": {}}
    manifest = _load_json(run_dir / "manifest.json")
    summary = _load_json(run_dir / "workflow_summary.json")
    for scene in summary["scenes"]:
        report_index["scenes"].append(scene["scene_name"])
    report_index["parameters"] = manifest["parameters"]

    # 3. 收集每個階段的視覺化輸出
    for stage in summary["stages"]:
        stage_name = stage["name"]
        report_index["stages"][stage_name] = {"outputs": []}
        stage_dir = run_dir / stage_name
        if stage_dir.exists():
            for img_file in sorted(stage_dir.glob("*.png")):
                if img_file.name.startswith("compare_"):
                    # 對比圖單獨處理
                    report_index["stages"][stage_name]["outputs"].append(
                        {"type": "comparison", "path": str(img_file.relative_to(run_dir))}
                    )
                else:
                    # 其他圖像預覽
                    img = Image.open(img_file)
                    img.thumbnail((max_preview_width, max_preview_width))
                    img.save(output_dir / img_file.name)
                    report_index["stages"][stage_name]["outputs"].append(
                        {"type": "image", "path": str(img_file.relative_to(run_dir))}
                    )

    # 4. 儲存報告索引
    _save_json(report_index, output_dir / "report_index.json")


@click.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.argument("report_name", type=str)
@click.option("--max-preview-width", default=640, help="縮圖最大寬度")
def main(run_dir, report_name, max_preview_width):
    """CLI 入口，可對任一 run 產生報告"""
    build_report(run_dir, report_name, max_preview_width)
    click.echo(f"報告已儲存至 {run_dir}/report 目錄")


if __name__ == "__main__":
    main()
