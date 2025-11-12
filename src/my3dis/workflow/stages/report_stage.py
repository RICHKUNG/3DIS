"""Report generation stage runner."""

from __future__ import annotations

from pathlib import Path

from my3dis.generate_report import build_report
from my3dis.workflow.stage_runner import StageRunner
from my3dis.workflow.summary import StageRecorder, export_stage_timings


class ReportStageRunner(StageRunner):
    """Stage runner for report generation."""

    @property
    def name(self) -> str:
        return "report"

    def should_run(self) -> bool:
        stage_cfg = self._stage_cfg()
        return stage_cfg.get('enabled', True)

    def execute(self) -> None:
        """Generate markdown report and export timings."""
        stage_cfg = self._stage_cfg()

        run_dir = getattr(self.context, '_run_dir', None)
        if run_dir is None:
            print('Stage Report: No run directory available, skipping')
            return

        run_dir_path = Path(run_dir)
        report_name = stage_cfg.get('name', 'report.md')
        max_width = int(stage_cfg.get('max_width', 960))

        print('Stage Report: 生成 Markdown 紀錄')
        with StageRecorder(self.summary, 'report', self._stage_gpu_env):
            build_report(run_dir_path, report_name=report_name, max_preview_width=max_width)
            report_summary = self._stage_summary().setdefault('params', {})
            report_summary['max_width'] = max_width
            report_summary['report_name'] = report_name

        # Export timings
        record_timings_flag = bool(stage_cfg.get('record_timings'))
        timing_output_name = stage_cfg.get('timing_output')
        if record_timings_flag or timing_output_name:
            output_name = timing_output_name or 'stage_timings.json'
            timings_path = run_dir_path / output_name
            export_stage_timings(self.summary, timings_path)
            artifacts_entry = self._stage_summary().setdefault('artifacts', {})
            artifacts_entry['timings'] = str(timings_path)

        # Generate experiment-level check report
        generate_check_report = bool(stage_cfg.get('generate_check_report', False))
        if generate_check_report and run_dir_path.parent.is_dir():
            try:
                from my3dis.workflow.reporting import generate_experiment_check_report
                check_path = generate_experiment_check_report(run_dir_path.parent)
                artifacts_entry = self._stage_summary().setdefault('artifacts', {})
                artifacts_entry['experiment_check'] = str(check_path.relative_to(run_dir_path.parent))
                print(f'Generated experiment check report: {check_path}')
            except Exception as exc:
                import logging
                LOGGER = logging.getLogger(__name__)
                LOGGER.warning("Failed to generate experiment check report: %s", exc)
