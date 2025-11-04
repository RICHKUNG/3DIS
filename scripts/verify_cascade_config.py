#!/usr/bin/env python3
"""
Quick verification script for cascade filtering configuration.
Validates YAML config and parameter passing without running the full pipeline.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import yaml


def verify_yaml_config(config_path):
    """Verify YAML configuration contains cascade filtering settings."""
    print(f"Checking config: {config_path}")
    print("-" * 60)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Check experiment section
    exp_cfg = cfg.get('experiment', {})
    print(f"✓ Experiment name: {exp_cfg.get('name', 'N/A')}")
    print(f"✓ Scenes: {exp_cfg.get('scenes', 'N/A')}")
    print(f"✓ Levels: {exp_cfg.get('levels', 'N/A')}")

    # Check SSAM stage
    ssam_cfg = cfg.get('stages', {}).get('ssam', {})
    print(f"\nSSAM Stage Configuration:")
    print(f"  enabled: {ssam_cfg.get('enabled', False)}")
    print(f"  min_area: {ssam_cfg.get('min_area', 'N/A')}")
    print(f"  stability_threshold: {ssam_cfg.get('stability_threshold', 'N/A')}")

    # Key parameter for orphan fix
    cascade_filtering = ssam_cfg.get('cascade_filtering', None)
    if cascade_filtering is None:
        print(f"  ⚠ cascade_filtering: NOT SET (will use default: true)")
    elif cascade_filtering:
        print(f"  ✓ cascade_filtering: true (ENABLED)")
    else:
        print(f"  ⚠ cascade_filtering: false (DISABLED)")

    # Check tracker stage
    tracker_cfg = cfg.get('stages', {}).get('tracker', {})
    print(f"\nTracker Stage Configuration:")
    print(f"  enabled: {tracker_cfg.get('enabled', False)}")
    print(f"  visualize_families: {tracker_cfg.get('visualize_families', False)}")

    family_viz = tracker_cfg.get('family_viz', {})
    if family_viz:
        print(f"  family_viz.num_families: {family_viz.get('num_families', 'N/A')}")
        print(f"  family_viz.max_frames_per_family: {family_viz.get('max_frames_per_family', 'N/A')}")

    print()
    return cascade_filtering


def verify_parameter_chain():
    """Verify that parameter passing chain is complete."""
    print("Verifying parameter passing chain...")
    print("-" * 60)

    try:
        # Check stage_config.py
        from my3dis.workflow.stage_config import SSAMStageConfig

        # Check if cascade_filtering field exists
        import dataclasses
        fields = {f.name for f in dataclasses.fields(SSAMStageConfig)}

        if 'cascade_filtering' in fields:
            print("✓ SSAMStageConfig has 'cascade_filtering' field")
        else:
            print("✗ SSAMStageConfig missing 'cascade_filtering' field")
            return False

        # Check generate_candidates.py
        from my3dis.generate_candidates import run_generation
        import inspect
        sig = inspect.signature(run_generation)

        if 'cascade_filtering' in sig.parameters:
            print("✓ run_generation() has 'cascade_filtering' parameter")
        else:
            print("✗ run_generation() missing 'cascade_filtering' parameter")
            return False

        # Check ssam_progressive_adapter.py
        from my3dis.ssam_progressive_adapter import generate_with_progressive
        sig = inspect.signature(generate_with_progressive)

        if 'cascade_filtering' in sig.parameters:
            print("✓ generate_with_progressive() has 'cascade_filtering' parameter")
        else:
            print("✗ generate_with_progressive() missing 'cascade_filtering' parameter")
            return False

        # Check semantic_refinement.py
        from my3dis.semantic_refinement import progressive_refinement_masks
        sig = inspect.signature(progressive_refinement_masks)

        if 'cascade_filtering' in sig.parameters:
            print("✓ progressive_refinement_masks() has 'cascade_filtering' parameter")
        else:
            print("✗ progressive_refinement_masks() missing 'cascade_filtering' parameter")
            return False

        # Check cascade_filter module
        from my3dis.cascade_filter import CascadeFilter
        print("✓ CascadeFilter class imported successfully")

        print("\n✓ Complete parameter passing chain verified!")
        return True

    except Exception as e:
        print(f"✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test loading a config through the workflow system."""
    print("\nTesting config loading through workflow system...")
    print("-" * 60)

    try:
        # Import after verifying imports work
        from my3dis.workflow.stage_config import SSAMStageConfig
        import yaml

        # Load test config
        config_path = project_root / 'configs' / 'multiscan' / 'test_orphan_fix.yaml'
        if not config_path.exists():
            print(f"⚠ Test config not found: {config_path}")
            return True  # Not an error, just skip

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Try to create SSAMStageConfig
        exp_cfg = cfg.get('experiment', {})
        ssam_cfg = cfg.get('stages', {}).get('ssam', {})

        # Mock minimal required fields
        data_path = exp_cfg.get('dataset_root', '/tmp') + '/scene_00000_00/outputs/color'
        output_root = exp_cfg.get('output_root', '/tmp/output').format(name=exp_cfg.get('name', 'test'))

        try:
            stage_config = SSAMStageConfig.from_yaml_config(
                stage_cfg=ssam_cfg,
                experiment_cfg=exp_cfg,
                data_path=data_path,
                output_root=output_root,
            )

            print(f"✓ SSAMStageConfig loaded successfully")
            print(f"  cascade_filtering: {stage_config.cascade_filtering}")
            print(f"  stability_threshold: {stage_config.stability_threshold}")
            print(f"  min_area: {stage_config.min_area}")

            # Test to_legacy_kwargs
            kwargs = stage_config.to_legacy_kwargs()
            if 'cascade_filtering' in kwargs:
                print(f"✓ cascade_filtering in to_legacy_kwargs(): {kwargs['cascade_filtering']}")
            else:
                print(f"✗ cascade_filtering missing from to_legacy_kwargs()")
                return False

            return True

        except Exception as e:
            print(f"✗ Failed to create SSAMStageConfig: {e}")
            return False

    except Exception as e:
        print(f"✗ Error during config loading test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main verification routine."""
    print("=" * 60)
    print("Cascade Filtering Configuration Verification")
    print("=" * 60)
    print()

    all_pass = True

    # 1. Verify test config exists and is correct
    test_config = project_root / 'configs' / 'multiscan' / 'test_orphan_fix.yaml'
    if test_config.exists():
        cascade_enabled = verify_yaml_config(test_config)
        if not cascade_enabled:
            print("⚠ Warning: cascade_filtering not explicitly enabled in test config")
        print()
    else:
        print(f"⚠ Test config not found: {test_config}")
        print()

    # 2. Verify parameter passing chain
    if not verify_parameter_chain():
        all_pass = False
    print()

    # 3. Test config loading
    if not test_config_loading():
        all_pass = False
    print()

    # Summary
    print("=" * 60)
    if all_pass:
        print("✓ All verifications passed!")
        print("✓ Cascade filtering is properly integrated")
        print()
        print("Next steps:")
        print("1. Run unit tests: python3 scripts/test_cascade_filter_logic.py")
        print("2. Run full test: ./scripts/test_orphan_fix.sh")
        print("=" * 60)
        return 0
    else:
        print("✗ Some verifications failed")
        print("Please check the errors above")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
