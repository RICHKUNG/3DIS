#!/usr/bin/env python3
"""
Validate format versions and consistency across My3DIS experiment outputs.

Usage:
    python scripts/validate_formats.py --experiments-root outputs/experiments
    python scripts/validate_formats.py --experiment outputs/experiments/v4_246.../scene_00005_00
    python scripts/validate_formats.py --file outputs/.../relations.json --type relations

Author: Claude Code Assistant
Created: 2025-10-22
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from my3dis.format_utils import (
    detect_relations_format,
    detect_index_format,
    detect_npz_manifest_format,
    validate_format_v3,
    get_object_count,
)

LOGGER = logging.getLogger(__name__)


class ValidationReport:
    """Track validation results."""

    def __init__(self):
        self.total_files = 0
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors: List[Tuple[Path, str]] = []
        self.version_counts: Dict[str, int] = {}

    def add_pass(self, version: str):
        self.total_files += 1
        self.passed += 1
        self.version_counts[version] = self.version_counts.get(version, 0) + 1

    def add_fail(self, path: Path, error: str, version: str = 'unknown'):
        self.total_files += 1
        self.failed += 1
        self.errors.append((path, error))
        self.version_counts[version] = self.version_counts.get(version, 0) + 1

    def add_warning(self):
        self.warnings += 1

    def print_summary(self):
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total files checked: {self.total_files}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"⚠️  Warnings: {self.warnings}")

        if self.version_counts:
            print("\nVersion Distribution:")
            for version in sorted(self.version_counts.keys()):
                count = self.version_counts[version]
                pct = (count / self.total_files * 100) if self.total_files > 0 else 0
                print(f"  {version}: {count} files ({pct:.1f}%)")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for path, error in self.errors[:10]:  # Show first 10
                print(f"  ❌ {path.name}: {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")

        print("=" * 70)


def validate_relations_file(path: Path, report: ValidationReport, strict: bool = False) -> bool:
    """Validate a single relations.json file."""
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        version = detect_relations_format(data)
        obj_count = get_object_count(data)

        LOGGER.debug("Checking %s: version=%s, objects=%d", path, version, obj_count)

        # Check for v3.0 compliance if strict mode
        if strict and version != "3.0":
            report.add_warning()
            LOGGER.warning("%s: Not v3.0 format (found: %s)", path, version)

        # Validate v3.0 structure if applicable
        if version == "3.0":
            is_valid = validate_format_v3(data, 'relations')
            if not is_valid:
                report.add_fail(path, f"Invalid v3.0 structure", version)
                return False

        # Check for reasonable object count
        if obj_count == 0:
            report.add_warning()
            LOGGER.warning("%s: 0 objects detected", path)

        report.add_pass(version)
        return True

    except json.JSONDecodeError as e:
        report.add_fail(path, f"JSON decode error: {e}", 'invalid')
        return False
    except Exception as e:
        report.add_fail(path, f"Unexpected error: {e}", 'error')
        return False


def validate_index_file(path: Path, report: ValidationReport, strict: bool = False) -> bool:
    """Validate a single index.json file."""
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        version = detect_index_format(data)
        LOGGER.debug("Checking %s: version=%s", path, version)

        if strict and version != "3.0":
            report.add_warning()
            LOGGER.warning("%s: Not v3.0 format (found: %s)", path, version)

        if version == "3.0":
            is_valid = validate_format_v3(data, 'index')
            if not is_valid:
                report.add_fail(path, f"Invalid v3.0 structure", version)
                return False

        report.add_pass(version)
        return True

    except json.JSONDecodeError as e:
        report.add_fail(path, f"JSON decode error: {e}", 'invalid')
        return False
    except Exception as e:
        report.add_fail(path, f"Unexpected error: {e}", 'error')
        return False


def validate_experiment(exp_dir: Path, report: ValidationReport, strict: bool = False):
    """Validate all format files in an experiment directory."""
    # Check relations.json at experiment root
    relations_path = exp_dir / 'relations.json'
    if relations_path.exists():
        validate_relations_file(relations_path, report, strict)
    else:
        LOGGER.warning("No relations.json found in %s", exp_dir)

    # Check per-level files
    for level_dir in exp_dir.glob('level_*'):
        # Check index.json
        index_path = level_dir / 'index.json'
        if index_path.exists():
            validate_index_file(index_path, report, strict)

        # Check tree.json
        tree_path = level_dir / 'relations' / 'tree.json'
        if tree_path.exists():
            # Tree has similar structure to index
            validate_index_file(tree_path, report, strict)


def scan_experiments_root(experiments_root: Path, report: ValidationReport, strict: bool = False):
    """Scan all experiments in a root directory."""
    # Find all experiment patterns (v2_*, v4_*, etc.)
    for exp_pattern_dir in experiments_root.glob('v*_*'):
        if not exp_pattern_dir.is_dir():
            continue

        LOGGER.info("Scanning experiment group: %s", exp_pattern_dir.name)

        # Find scene directories
        for scene_dir in exp_pattern_dir.glob('scene_*'):
            if not scene_dir.is_dir():
                continue

            LOGGER.debug("  Validating scene: %s", scene_dir.name)
            validate_experiment(scene_dir, report, strict)


def main():
    parser = argparse.ArgumentParser(
        description="Validate My3DIS data format versions and consistency"
    )
    parser.add_argument(
        '--experiments-root',
        type=Path,
        help='Root directory containing experiments (validates all)',
    )
    parser.add_argument(
        '--experiment',
        type=Path,
        help='Single experiment directory to validate',
    )
    parser.add_argument(
        '--file',
        type=Path,
        help='Single file to validate',
    )
    parser.add_argument(
        '--type',
        choices=['relations', 'index', 'tree'],
        default='relations',
        help='File type when using --file',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Strict mode: warn if not v3.0 format',
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable debug logging',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    report = ValidationReport()

    # Determine mode
    if args.file:
        # Single file mode
        if not args.file.exists():
            LOGGER.error("File not found: %s", args.file)
            sys.exit(1)

        LOGGER.info("Validating single file: %s", args.file)
        if args.type in ('relations', 'tree'):
            validate_relations_file(args.file, report, args.strict)
        elif args.type == 'index':
            validate_index_file(args.file, report, args.strict)

    elif args.experiment:
        # Single experiment mode
        if not args.experiment.exists():
            LOGGER.error("Experiment directory not found: %s", args.experiment)
            sys.exit(1)

        LOGGER.info("Validating experiment: %s", args.experiment)
        validate_experiment(args.experiment, report, args.strict)

    elif args.experiments_root:
        # Batch mode
        if not args.experiments_root.exists():
            LOGGER.error("Experiments root not found: %s", args.experiments_root)
            sys.exit(1)

        LOGGER.info("Scanning experiments root: %s", args.experiments_root)
        scan_experiments_root(args.experiments_root, report, args.strict)

    else:
        parser.print_help()
        sys.exit(1)

    # Print summary
    report.print_summary()

    # Exit with error code if failures
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == '__main__':
    main()
