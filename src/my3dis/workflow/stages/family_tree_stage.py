"""Family tree generation stage runner."""

from __future__ import annotations

from pathlib import Path

from my3dis.workflow.stage_runner import StageRunner
from my3dis.workflow.summary import StageRecorder


class FamilyTreeStageRunner(StageRunner):
    """Stage runner for family tree generation."""

    @property
    def name(self) -> str:
        return "family_tree"

    def should_run(self) -> bool:
        # Family tree runs if tracker is enabled
        tracker_cfg = self.context.stages_cfg.get('tracker', {})
        return tracker_cfg.get('enabled', True)

    def execute(self) -> None:
        """Generate unified family tree from per-level provenance trees."""
        run_dir = getattr(self.context, '_run_dir', None)
        if run_dir is None:
            return

        run_dir_path = Path(run_dir)
        relations_dir = run_dir_path / 'relations'
        if not relations_dir.exists():
            return

        provenance_trees = list(relations_dir.glob('provenance_tree_L*.json'))
        if not provenance_trees:
            return

        print('Stage Family Tree: 生成統一家族樹')

        try:
            from my3dis.family_tree_builder import build_family_tree, save_family_tree

            levels = self.context.experiment_cfg.get('levels', [2, 4, 6])
            tracker_cfg = self.context.stages_cfg.get('tracker', {})
            mask_scale_ratio = tracker_cfg.get('downscale_ratio', 0.3)

            with StageRecorder(self.summary, 'family_tree', self._stage_gpu_env):
                family_tree = build_family_tree(
                    str(run_dir_path),
                    levels=levels,
                    mask_scale_ratio=mask_scale_ratio,
                )

                output_path = relations_dir / 'family_tree.json'
                save_family_tree(family_tree, str(output_path))

                stage_summary = self._stage_summary()
                stage_summary['artifacts'] = {
                    'family_tree': str(output_path.relative_to(run_dir_path))
                }
                stage_summary['statistics'] = family_tree.get('statistics', {})

                print(f'  ✓ Family tree saved: {output_path.relative_to(run_dir_path)}')
                stats = family_tree.get('statistics', {})
                print(f'    - Total objects: {stats.get("total_objects", 0)}')
                print(f'    - Total families: {stats.get("total_families", 0)}')
                print(f'    - Orphans: {stats.get("orphan_count", 0)}')

        except Exception as e:
            print(f'  Warning: Failed to build family tree: {e}')
            stage_summary = self._stage_summary()
            stage_summary.setdefault('warnings', []).append({
                'message': f'Failed to build family tree: {e}',
                'stage': 'family_tree',
            })
