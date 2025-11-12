"""Family visualization stage runner."""

from __future__ import annotations

from pathlib import Path

from my3dis.workflow.stage_runner import StageRunner
from my3dis.workflow.summary import StageRecorder


class FamilyVizStageRunner(StageRunner):
    """Stage runner for family visualization."""

    @property
    def name(self) -> str:
        return "family_viz"

    def should_run(self) -> bool:
        tracker_cfg = self.context.stages_cfg.get('tracker', {})
        return tracker_cfg.get('enabled', True) and tracker_cfg.get('visualize_families', False)

    def execute(self) -> None:
        """Generate family visualizations (side-by-side L2/L4/L6 rendering)."""
        run_dir = getattr(self.context, '_run_dir', None)
        if run_dir is None:
            return

        run_dir_path = Path(run_dir)
        family_tree_path = run_dir_path / 'relations' / 'family_tree.json'

        if not family_tree_path.exists():
            print('  Warning: family_tree.json not found, skipping visualization')
            return

        print('Stage Family Visualization: 生成家族比較圖')

        try:
            import random
            import sys
            from pathlib import Path as P

            scripts_path = P(__file__).parent.parent.parent.parent.parent / 'scripts'
            if str(scripts_path) not in sys.path:
                sys.path.insert(0, str(scripts_path))

            from visualize_families import FamilyTreeQuery, visualize_family

            tracker_cfg = self.context.stages_cfg.get('tracker', {})
            viz_cfg = tracker_cfg.get('family_viz', {})
            num_families = viz_cfg.get('num_families', 10)
            max_frames = viz_cfg.get('max_frames_per_family', 3)
            output_subdir = viz_cfg.get('output_subdir', 'visualizations')
            random_seed = viz_cfg.get('random_seed', 42)
            min_levels = viz_cfg.get('min_levels', 2)

            output_dir = run_dir_path / output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)

            with StageRecorder(self.summary, 'family_viz', self._stage_gpu_env):
                query = FamilyTreeQuery(str(family_tree_path))

                families = query.tree.get('families', [])
                valid_families = [
                    family for family in families
                    if len(family.get('levels', {})) >= min_levels
                ]

                if not valid_families:
                    print(f'  No families with >= {min_levels} levels found')
                    stage_summary = self._stage_summary()
                    stage_summary['status'] = 'skipped'
                    stage_summary['reason'] = f'No families with >= {min_levels} levels'
                    return

                print(f'  Found {len(valid_families)} families with >= {min_levels} levels')

                random.seed(random_seed)
                num_to_sample = min(num_families, len(valid_families)) if num_families else len(valid_families)
                selected_families = random.sample(valid_families, num_to_sample)

                print(f'  Visualizing {num_to_sample} families...')

                total_images = 0
                for i, family in enumerate(selected_families, 1):
                    family_members = family['members']
                    output_paths = visualize_family(
                        query,
                        family_members,
                        self.context.data_path,
                        str(output_dir),
                        i,
                        max_frames,
                    )
                    total_images += len(output_paths)

                stage_summary = self._stage_summary()
                stage_summary['artifacts'] = {
                    'visualization_dir': str(output_dir.relative_to(run_dir_path))
                }
                stage_summary['statistics'] = {
                    'families_visualized': num_to_sample,
                    'total_images': total_images,
                    'families_available': len(valid_families),
                }
                stage_summary['params'] = {
                    'num_families': num_families,
                    'max_frames_per_family': max_frames,
                    'min_levels': min_levels,
                    'random_seed': random_seed,
                }

                print(f'  ✓ Generated {total_images} visualization images')
                print(f'    Output: {output_dir.relative_to(run_dir_path)}')

        except ImportError as e:
            print(f'  Warning: Failed to import visualization module: {e}')
            stage_summary = self._stage_summary()
            stage_summary.setdefault('warnings', []).append({
                'message': f'Import error: {e}',
                'stage': 'family_viz',
            })
        except Exception as e:
            print(f'  Warning: Failed to generate visualizations: {e}')
            stage_summary = self._stage_summary()
            stage_summary.setdefault('warnings', []).append({
                'message': f'Visualization error: {e}',
                'stage': 'family_viz',
            })
