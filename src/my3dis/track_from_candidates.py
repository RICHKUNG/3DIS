"""
Tracking-only stage: read filtered candidates saved by generate_candidates.py
and run SAM2 masklet propagation. Designed to run in a SAM2-capable env.
"""
# Ensure src/ is in path for direct execution (inline to avoid circular import)
if __package__ is None or __package__ == '':
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

from my3dis.common_utils import ensure_dir, setup_logging
from my3dis.pipeline_defaults import (
    DEFAULT_SAM2_CFG as _DEFAULT_SAM2_CFG_PATH,
    DEFAULT_SAM2_CKPT as _DEFAULT_SAM2_CKPT_PATH,
    DEFAULT_SAM2_ROOT as _DEFAULT_SAM2_ROOT,
    expand_default,
)
from my3dis.tracking import TimingAggregator, format_duration_precise
from my3dis.tracking.level_runner import run_level_tracking
from my3dis.tracking.pipeline_context import (
    LevelRunResult,
    TrackingContext,
    ensure_subset_video,
    prepare_tracking_context,
    resolve_long_tail_area_threshold,
    update_manifest,
)

_SAM2_ROOT_STR = expand_default(_DEFAULT_SAM2_ROOT)
if _SAM2_ROOT_STR not in sys.path:
    sys.path.insert(0, _SAM2_ROOT_STR)

from sam2.build_sam import build_sam2_video_predictor

LOGGER = logging.getLogger("my3dis.track_from_candidates")

DEFAULT_SAM2_ROOT = _SAM2_ROOT_STR
DEFAULT_SAM2_CFG = str(_DEFAULT_SAM2_CFG_PATH)
DEFAULT_SAM2_CKPT = str(_DEFAULT_SAM2_CKPT_PATH)


def resolve_sam2_config_path(config_arg: str) -> str:
    cfg_path = os.path.expanduser(config_arg)
    if os.path.isfile(cfg_path):
        base = os.path.join(DEFAULT_SAM2_ROOT, 'sam2')
        rel = os.path.relpath(cfg_path, base)
        rel = rel.replace(os.sep, '/')
        if rel.endswith('.yaml'):
            rel = rel[:-5]
        return rel
    return config_arg


def configure_logging(explicit_level: Optional[int] = None) -> int:
    level = setup_logging(explicit_level=explicit_level)
    LOGGER.setLevel(level)
    return level


def run_tracking(
    *,
    data_path: str,
    candidates_root: str,
    output: str,
    levels: Union[str, List[int]] = "2,4,6",
    sam2_cfg: Optional[Union[str, os.PathLike]] = DEFAULT_SAM2_CFG,
    sam2_ckpt: Optional[Union[str, os.PathLike]] = DEFAULT_SAM2_CKPT,
    sam2_max_propagate: Optional[int] = None,
    log_level: Optional[int] = None,
    iou_threshold: float = 0.6,
    long_tail_box_prompt: bool = False,
    all_box_prompt: bool = False,
    mask_scale_ratio: float = 1.0,
    comparison_sample_stride: Optional[int] = None,
    comparison_max_samples: Optional[int] = None,
    render_viz: bool = True,
) -> str:
    if not sam2_cfg:
        sam2_cfg = DEFAULT_SAM2_CFG
    if not sam2_ckpt:
        sam2_ckpt = DEFAULT_SAM2_CKPT
    sam2_cfg = os.fspath(sam2_cfg) if isinstance(sam2_cfg, os.PathLike) else sam2_cfg
    sam2_ckpt = os.fspath(sam2_ckpt) if isinstance(sam2_ckpt, os.PathLike) else sam2_ckpt

    configure_logging(log_level)

    try:
        mask_scale_ratio = float(mask_scale_ratio)
    except (TypeError, ValueError):
        raise ValueError(f'Invalid mask_scale_ratio={mask_scale_ratio!r}')
    if mask_scale_ratio <= 0.0 or mask_scale_ratio > 1.0:
        raise ValueError('mask_scale_ratio must be within (0, 1]')

    overall_start = time.perf_counter()
    if isinstance(levels, str):
        level_list = [int(x) for x in levels.split(',') if x.strip()]
    else:
        level_list = [int(x) for x in levels]

    LOGGER.info("SAM2 tracking started (levels=%s)", ",".join(str(x) for x in level_list))

    context = prepare_tracking_context(
        candidates_root=candidates_root,
        level_list=level_list,
        sam2_max_propagate=sam2_max_propagate,
    )
    manifest = context.manifest
    level_list = context.level_list
    ssam_frames = context.ssam_frames
    ssam_absolute_indices = context.ssam_absolute_indices
    ssam_local_indices = context.ssam_local_indices
    ssam_freq = context.ssam_freq
    sam2_max_propagate = context.sam2_max_propagate

    LOGGER.info(
        "Configuration: ssam_freq=%d, sam2_max_propagate=%s, ssam_frames=%d, iou_threshold=%.2f, mask_scale_ratio=%.3f, preview_stride=%s, preview_max=%s, render_viz=%s",
        ssam_freq,
        sam2_max_propagate,
        len(ssam_frames),
        iou_threshold,
        mask_scale_ratio,
        comparison_sample_stride,
        comparison_max_samples,
        render_viz,
    )

    long_tail_area_threshold = resolve_long_tail_area_threshold(
        manifest=manifest,
        long_tail_box_prompt=long_tail_box_prompt,
        all_box_prompt=all_box_prompt,
    )

    try:
        os.chdir(DEFAULT_SAM2_ROOT)
    except Exception:
        pass
    sam2_cfg_resolved = resolve_sam2_config_path(sam2_cfg)
    # predictor = build_sam2_video_predictor(sam2_cfg_resolved, sam2_ckpt, vos_optimized=True)
    # print("我有用VOS版本的SAM2了！")
    predictor = build_sam2_video_predictor(sam2_cfg_resolved, sam2_ckpt)

    out_root = ensure_dir(output)
    subset_dir, subset_map = ensure_subset_video(
        context,
        data_path=data_path,
        out_root=out_root,
    )
    LOGGER.info("Selected frames available at %s", subset_dir)

    frame_index_to_name = {
        int(idx): str(name)
        for idx, name in zip(context.selected_indices, context.selected_frames)
    }

    level_results: List[LevelRunResult] = []
    overall_timer = TimingAggregator()

    for level in level_list:
        result = run_level_tracking(
            level=level,
            candidates_root=candidates_root,
            data_path=data_path,
            subset_dir=subset_dir,
            subset_map=subset_map,
            predictor=predictor,
            frame_index_lookup=frame_index_to_name,
            selected_indices=context.selected_indices,
            ssam_local_indices=ssam_local_indices,
            sam2_max_propagate=sam2_max_propagate,
            iou_threshold=iou_threshold,
            long_tail_box_prompt=long_tail_box_prompt,
            all_box_prompt=all_box_prompt,
            long_tail_area_threshold=long_tail_area_threshold,
            mask_scale_ratio=mask_scale_ratio,
            comparison_sample_stride=comparison_sample_stride,
            comparison_max_samples=comparison_max_samples,
            render_viz=render_viz,
            out_root=out_root,
        )
        level_results.append(result)
        overall_timer.merge(result.timer)

    if level_results:
        summary = "; ".join(
            f"L{lvl}: {objs} objects / {frames} frames "
            f"(track={format_duration_precise(track)}, persist={format_duration_precise(persist)}, "
            f"viz={format_duration_precise(viz)}, render={format_duration_precise(render)}, "
            f"total={format_duration_precise(total)})"
            for (
                lvl,
                objs,
                frames,
                track,
                persist,
                viz,
                render,
                total,
            ) in (result.stats for result in level_results)
        )
        LOGGER.info("Tracking summary → %s", summary)

    if overall_timer.items():
        category_summary = []
        for label, prefix in [
            ("track", 'track.'),
            ("persist", 'persist.'),
            ("viz", 'viz.'),
        ]:
            total = overall_timer.total_prefix(prefix)
            if total > 0:
                category_summary.append(f"{label}={format_duration_precise(total)}")
        if category_summary:
            LOGGER.info("Aggregate timing by stage → %s", ", ".join(category_summary))
        LOGGER.debug("Aggregate timing breakdown → %s", overall_timer.format_breakdown())

    update_manifest(
        context,
        out_root=out_root,
        level_results=level_results,
        mask_scale_ratio=mask_scale_ratio,
        render_viz=render_viz,
    )

    LOGGER.info("Tracking results saved at %s", out_root)
    LOGGER.info(
        "Tracking completed in %s",
        format_duration_precise(time.perf_counter() - overall_start),
    )

    return out_root


def main():
    ap = argparse.ArgumentParser(description="SAM2 tracking from pre-generated candidates")
    ap.add_argument('--data-path', required=True, help='Original frames dir')
    ap.add_argument('--candidates-root', required=True, help='Root containing level_*/filtered')
    ap.add_argument('--sam2-cfg', default=DEFAULT_SAM2_CFG,
                    help='SAM2 config YAML or Hydra path (default: sam2.1_hiera_l)')
    ap.add_argument('--sam2-ckpt', default=DEFAULT_SAM2_CKPT,
                    help='SAM2 checkpoint path (default: sam2.1_hiera_large.pt)')
    ap.add_argument('--output', required=True)
    ap.add_argument('--levels', default='2,4,6')
    ap.add_argument('--sam2-max-propagate', type=int, default=None,
                    help='Limit SAM2 propagation to N frames per direction (default: unlimited)')
    ap.add_argument('--iou-threshold', type=float, default=0.6,
                    help='IoU threshold for deduplicating SAM2 prompts (default: 0.6)')
    ap.add_argument('--long-tail-box-prompt', action='store_true',
                    help='Convert long-tail small objects to SAM2 box prompts')
    ap.add_argument('--all-box-prompt', action='store_true',
                    help='Convert all mask prompts to SAM2 box prompts')
    ap.add_argument('--mask-scale-ratio', type=float, default=1.0,
                    help='Downscale masks before persistence (e.g., 0.3 keeps 30% resolution)')
    ap.add_argument('--skip-viz', action='store_true',
                    help='Disable all additional visualization renders to keep outputs minimal')
    args = ap.parse_args()

    run_tracking(
        data_path=args.data_path,
        candidates_root=args.candidates_root,
        output=args.output,
        levels=args.levels,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        sam2_max_propagate=args.sam2_max_propagate,
        iou_threshold=args.iou_threshold,
        long_tail_box_prompt=args.long_tail_box_prompt,
        all_box_prompt=args.all_box_prompt,
        mask_scale_ratio=args.mask_scale_ratio,
        render_viz=not args.skip_viz,
    )


if __name__ == '__main__':
    main()
