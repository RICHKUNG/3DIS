#!/usr/bin/env python3
"""Generate Markdown summary for a My3DIS experiment run."""
from __future__ import annotations

if __package__ is None or __package__ == '':
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))




import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from my3dis.common_utils import ensure_dir, format_duration


@dataclass
class StageTiming:
    name: str
    duration_sec: float
    start: Optional[str] = None
    end: Optional[str] = None
    gpu: Optional[str] = None

    @property
    def duration_text(self) -> str:
        return format_duration(self.duration_sec)


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open('r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def collect_stage_timings(summary: Optional[dict]) -> List[StageTiming]:
    if not summary:
        return []
    stages = summary.get('stages', {})
    timings: List[StageTiming] = []
    for name, meta in stages.items():
        duration = float(meta.get('duration_sec', 0.0))
        timings.append(
            StageTiming(
                name=name,
                duration_sec=duration,
                start=meta.get('started_at'),
                end=meta.get('ended_at'),
                gpu=meta.get('gpu'),
            )
        )
    # Preserve order defined in summary['order'] if available
    order = summary.get('order') if summary else None
    if isinstance(order, list):
        ordering = {name: idx for idx, name in enumerate(order)}
        timings.sort(key=lambda t: ordering.get(t.name, len(order) + 1))
    else:
        timings.sort(key=lambda t: t.name)
    return timings


def pick_first_mid_last(items: Sequence[Tuple[int, Path]]) -> List[Tuple[str, int, Path]]:
    if not items:
        return []
    count = len(items)
    selected: List[Tuple[str, int, Path]] = []
    indices = [0, count // 2, count - 1]
    labels = ['第一張', '中位張', '最後一張']
    used_positions = set()
    for label, idx in zip(labels, indices):
        idx = max(0, min(count - 1, idx))
        if idx in used_positions:
            continue
        used_positions.add(idx)
        selected.append((label, items[idx][0], items[idx][1]))
    return selected


def downscale_image(src: Path, dst: Path, max_width: int) -> None:
    with Image.open(src) as img:
        width, height = img.size
        if width <= max_width:
            ensure_dir(dst.parent)
            img.save(dst)
            return
        scale = max_width / float(width)
        new_size = (int(width * scale), int(height * scale))
        ensure_dir(dst.parent)
        img.resize(new_size, resample=Image.BILINEAR).save(dst)


def render_level_section(
    *,
    level: int,
    viz_dir: Path,
    report_dir: Path,
    max_width: int,
    markdown_root: Path,
) -> List[str]:
    compare_dir = viz_dir / 'compare'
    if not compare_dir.exists():
        return [f"### Level {level}", '', '無可用的比較圖片。', '']

    entries: List[Tuple[int, Path]] = []
    for path in sorted(compare_dir.glob('frame_*_L*.png')):
        stem = path.stem
        parts = stem.split('_')
        try:
            frame_idx = int(parts[1])
        except (IndexError, ValueError):
            continue
        entries.append((frame_idx, path))
    entries.sort(key=lambda item: item[0])
    selections = pick_first_mid_last(entries)
    lines = [f"### Level {level}", '']
    if not selections:
        lines.append('無可用的比較圖片。')
        lines.append('')
        return lines

    lines.append('| 代表幀 | 預覽 |')
    lines.append('| --- | --- |')

    for label, frame_idx, src_path in selections:
        safe_label = label
        dst_path = report_dir / f'frame_{frame_idx:05d}_{label}.png'
        downscale_image(src_path, dst_path, max_width=max_width)
        try:
            rel_path = dst_path.relative_to(markdown_root)
        except ValueError:
            rel_path = Path(os.path.relpath(dst_path, markdown_root))
        lines.append(
            f"| {safe_label} (frame {frame_idx:05d}) | ![]({rel_path.as_posix()}) |"
        )

    lines.append('')
    return lines


def build_report(
    run_dir: Path,
    *,
    report_name: str = 'report.md',
    max_preview_width: int = 960,
) -> Path:
    manifest = load_json(run_dir / 'manifest.json')
    summary = load_json(run_dir / 'workflow_summary.json')
    timings = collect_stage_timings(summary)

    report_path = run_dir / report_name
    timestamp = None
    if summary and summary.get('generated_at'):
        try:
            timestamp = datetime.fromisoformat(summary['generated_at']).strftime('%Y-%m-%d %H:%M')
        except ValueError:
            timestamp = summary['generated_at']

    lines: List[str] = []
    run_name = run_dir.name
    lines.append(f"# 報告：{run_name}")
    if timestamp:
        lines.append(f"生成時間：{timestamp}")
    lines.append('')

    if summary and summary.get('config_snapshot'):
        cfg = summary['config_snapshot']
        exp = cfg.get('experiment', {}) if isinstance(cfg, dict) else {}
        scene = exp.get('name') or exp.get('scene')
        data_path = exp.get('data_path')
        if scene or data_path:
            meta_line = '實驗設定：'
            parts = []
            if scene:
                parts.append(f"場景 `{scene}`")
            if data_path:
                parts.append(f"影像來源 `{data_path}`")
            if parts:
                lines.append(meta_line + '、'.join(parts))
                lines.append('')

    if timings:
        lines.append('## 執行時間概覽')
        lines.append('')
        lines.append('| Stage | GPU | 時間 |')
        lines.append('| --- | --- | --- |')
        for timing in timings:
            gpu = timing.gpu if timing.gpu is not None else '-'
            lines.append(f"| {timing.name} | {gpu} | {timing.duration_text} |")
        total = format_duration(sum(t.duration_sec for t in timings))
        lines.append(f"| 合計 |  | {total} |")
        lines.append('')

    if manifest:
        lines.append('## 參數摘要')
        lines.append('')
        ssam_levels = manifest.get('levels')
        frames = manifest.get('frames')
        ssam_freq = manifest.get('ssam_freq')
        filtering = manifest.get('filtering', {})
        lines.append('- Semantic-SAM 層級：' + (', '.join(str(x) for x in ssam_levels) if ssam_levels else '未記錄'))
        if isinstance(frames, str):
            lines.append(f'- 取樣範圍：`{frames}`')
        if ssam_freq is not None:
            lines.append(f'- SSAM 執行頻率：每 {ssam_freq} 幀一次')
        if filtering:
            applied = filtering.get('applied')
            min_area = filtering.get('min_area')
            stab = filtering.get('stability_threshold')
            if applied:
                lines.append(
                    f"- 遮罩篩選：min_area={min_area}, stability>={stab}"
                )
        lines.append('')

    levels = []
    if manifest and isinstance(manifest.get('levels'), list):
        try:
            levels = [int(x) for x in manifest['levels']]
        except (ValueError, TypeError):
            levels = []

    for level in levels:
        viz_dir = run_dir / f'level_{level}' / 'viz'
        report_dir = run_dir / f'level_{level}' / 'report'
        lines.extend(
            render_level_section(
                level=level,
                viz_dir=viz_dir,
                report_dir=report_dir,
                max_width=max_preview_width,
                markdown_root=run_dir,
            )
        )

    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate Markdown report for a workflow run')
    parser.add_argument('--run-dir', required=True, help='Path to the workflow run directory')
    parser.add_argument('--report-name', default='report.md', help='Markdown filename (default: report.md)')
    parser.add_argument('--max-width', type=int, default=960, help='Maximum width for preview images (default: 960)')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        print(f'Run directory {run_dir} does not exist', file=os.sys.stderr)
        return 1
    report_path = build_report(
        run_dir,
        report_name=args.report_name,
        max_preview_width=args.max_width,
    )
    print(f'Report written to {report_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
