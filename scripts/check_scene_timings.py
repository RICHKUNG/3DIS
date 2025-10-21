#!/usr/bin/env python3
"""Aggregate stage timings for experiment scenes and emit a markdown report."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# Update this list to change the default experiment directories processed when no arguments are provided.
DEFAULT_BASES = ["/media/Pluto/richkung/My3DIS/outputs/experiments/v2_135_ssam2_filter2k_fill10k_prop30_iou06_ds03",
                "/media/Pluto/richkung/My3DIS/outputs/experiments/v2_135_ssam2_filter2k_fill10k_prop50_iou06_ds03",
                "/media/Pluto/richkung/My3DIS/outputs/experiments/v2_135_ssam2_filter500_fill3k_prop30_iou06_ds03",
                "/media/Pluto/richkung/My3DIS/outputs/experiments/v2_246_ssam2_filter2k_fill10k_prop10_iou06_ds03",
                "/media/Pluto/richkung/My3DIS/outputs/experiments/v2_246_ssam2_filter2k_fill10k_prop30_iou05_ds03",
                "/media/Pluto/richkung/My3DIS/outputs/experiments/v2_246_ssam2_filter2k_fill10k_prop30_iou06_ds03",
                "/media/Pluto/richkung/My3DIS/outputs/experiments/v2_246_ssam2_filter2k_fill10k_prop50_iou06_ds03",
                "/media/Pluto/richkung/My3DIS/outputs/experiments/v2_246_ssam2_filter500_fill3k_prop30_iou06_ds03",
                "/media/Pluto/richkung/My3DIS/outputs/experiments/v2_246_ssam2_filter500_fill3k_prop30_iou07_ds03"]


@dataclass
class SceneTiming:
    """Container for per-scene timing information."""

    name: str
    summary_found: bool
    total: str
    stage_durations: Dict[str, str]


def format_seconds(seconds: Optional[float]) -> str:
    """Return a compact H:MM:SS or M:SS string for a duration in seconds."""
    if seconds is None:
        return "N/A"
    try:
        total_seconds = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "N/A"

    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def load_scene_timing(scene_dir: Path) -> SceneTiming:
    """Load timing information from summary.json if it exists."""
    summary_path = scene_dir / "summary.json"
    if not summary_path.is_file():
        return SceneTiming(scene_dir.name, False, "N/A", {})

    try:
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to read {summary_path}: {exc}", file=sys.stderr)
        return SceneTiming(scene_dir.name, True, "N/A", {})

    timings = summary.get("timings") or {}
    total = timings.get("total_duration_text") or format_seconds(
        timings.get("total_duration_sec")
    )

    stage_durations: Dict[str, str] = {}
    for stage in timings.get("stages") or []:
        stage_name = stage.get("stage") or "unknown"
        duration = stage.get("duration_text") or format_seconds(
            stage.get("duration_sec")
        )
        stage_durations[stage_name] = duration or "N/A"

    return SceneTiming(scene_dir.name, True, total or "N/A", stage_durations)


def collect_stage_order(scenes: Iterable[SceneTiming]) -> List[str]:
    """Derive a stable stage order from the scene data."""
    ordered: List[str] = []
    for scene in scenes:
        for stage in scene.stage_durations:
            if stage not in ordered:
                ordered.append(stage)
    return ordered


def render_markdown_table(stage_order: List[str], scenes: Iterable[SceneTiming]) -> str:
    """Produce a markdown table summarising timings."""
    headers = ["Scene", "Total"] + stage_order
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for scene in scenes:
        row = [scene.name, scene.total if scene.summary_found else "N/A"]
        for stage in stage_order:
            value = scene.stage_durations.get(stage, "N/A")
            row.append(value if scene.summary_found else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def build_report(base_path: Path) -> str:
    """Generate the markdown report for every scene under base_path."""
    scene_dirs = sorted(
        (path for path in base_path.iterdir() if path.is_dir()),
        key=lambda p: p.name,
    )

    scenes = [load_scene_timing(scene_dir) for scene_dir in scene_dirs]
    stage_order = collect_stage_order(scenes)

    return render_markdown_table(stage_order, scenes)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown table summarising scene stage timings."
    )
    parser.add_argument(
        "base_dirs",
        nargs="*",
        default=list(DEFAULT_BASES),
        help=(
            "One or more experiment directories containing scene folders "
            f"(default: {', '.join(DEFAULT_BASES)})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    exit_code = 0

    for base_dir in args.base_dirs:
        base_path = Path(base_dir).resolve()
        if not base_path.is_dir():
            print(f"[ERROR] Base directory not found: {base_path}", file=sys.stderr)
            exit_code = 1
            continue

        report = build_report(base_path)
        output_path = base_path / "check.md"
        output_path.write_text(report, encoding="utf-8")
        print(f"Wrote {output_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
