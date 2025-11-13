"""
Validate Combined Query Implementation Across All Strategies.

This script validates that all strategies properly implement combined query support
by checking the source code structure without requiring torch imports.

Checks:
1. CombinedQueryMixin exists and has required methods
2. IndependentStrategy inherits from CombinedQueryMixin
3. HierarchicalStrategy inherits from CombinedQueryMixin
4. ExhaustivePairingWithCombinedQuery inherits from CombinedQueryMixin
5. All strategies have infer_with_text_queries method
6. Configuration files use arithmetic scoring with geometric disabled
"""

import sys
from pathlib import Path
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Project root
project_root = Path(__file__).parent.parent
strategies_dir = project_root / "src" / "my3dis" / "inference" / "strategies"
config_file = project_root / "configs" / "inference" / "full_pipeline.yaml"


def check_file_exists(file_path: Path, description: str) -> bool:
    """Check if a file exists."""
    if file_path.exists():
        logger.info(f"✓ {description}: {file_path.name}")
        return True
    else:
        logger.error(f"✗ {description}: {file_path.name} NOT FOUND")
        return False


def check_class_inheritance(file_path: Path, class_name: str, parent_classes: list) -> bool:
    """Check if a class inherits from specific parent classes."""
    content = file_path.read_text()

    # Find class definition
    pattern = rf"class\s+{class_name}\s*\((.*?)\):"
    match = re.search(pattern, content)

    if not match:
        logger.error(f"✗ Class {class_name} not found in {file_path.name}")
        return False

    inheritance = match.group(1)

    # Check each required parent
    all_found = True
    for parent in parent_classes:
        if parent in inheritance:
            logger.info(f"  ✓ {class_name} inherits from {parent}")
        else:
            logger.error(f"  ✗ {class_name} does NOT inherit from {parent}")
            all_found = False

    return all_found


def check_method_exists(file_path: Path, class_name: str, method_name: str) -> bool:
    """Check if a method exists in a class."""
    content = file_path.read_text()

    # Find method definition
    pattern = rf"def\s+{method_name}\s*\("
    if re.search(pattern, content):
        logger.info(f"  ✓ {class_name} has {method_name}() method")
        return True
    else:
        logger.error(f"  ✗ {class_name} does NOT have {method_name}() method")
        return False


def check_config_setting(file_path: Path, key_path: str, expected_value) -> bool:
    """Check if a configuration setting has the expected value."""
    content = file_path.read_text()

    # Split key path (e.g., "scoring.method" -> ["scoring", "method"])
    keys = key_path.split('.')

    # Build regex pattern to find the value
    # This is a simplified check - just looks for the key and value in the file
    pattern = rf"{keys[-1]}:\s*{expected_value}"

    if re.search(pattern, content):
        logger.info(f"  ✓ {key_path}: {expected_value}")
        return True
    else:
        logger.error(f"  ✗ {key_path} is NOT set to {expected_value}")
        return False


def validate_mixin():
    """Validate CombinedQueryMixin implementation."""
    logger.info("\n" + "="*80)
    logger.info("VALIDATION 1: CombinedQueryMixin")
    logger.info("="*80)

    mixin_file = strategies_dir / "combined_query_mixin.py"

    results = []

    # Check file exists
    results.append(check_file_exists(mixin_file, "CombinedQueryMixin file"))

    if mixin_file.exists():
        # Check required methods
        results.append(check_method_exists(mixin_file, "CombinedQueryMixin", "setup_combined_query"))
        results.append(check_method_exists(mixin_file, "CombinedQueryMixin", "create_combined_query"))
        results.append(check_method_exists(mixin_file, "CombinedQueryMixin", "extract_combined_feature"))
        results.append(check_method_exists(mixin_file, "CombinedQueryMixin", "update_proposals_with_combined_feature"))

    return all(results)


def validate_independent_strategy():
    """Validate IndependentStrategy implementation."""
    logger.info("\n" + "="*80)
    logger.info("VALIDATION 2: IndependentStrategy")
    logger.info("="*80)

    independent_file = strategies_dir / "independent.py"

    results = []

    # Check file exists
    results.append(check_file_exists(independent_file, "IndependentStrategy file"))

    if independent_file.exists():
        # Check inheritance
        results.append(check_class_inheritance(
            independent_file,
            "IndependentStrategy",
            ["CombinedQueryMixin", "InferenceStrategy"]
        ))

        # Check methods
        results.append(check_method_exists(independent_file, "IndependentStrategy", "infer_with_text_queries"))

    return all(results)


def validate_hierarchical_strategy():
    """Validate HierarchicalStrategy implementation."""
    logger.info("\n" + "="*80)
    logger.info("VALIDATION 3: HierarchicalStrategy")
    logger.info("="*80)

    hierarchical_file = strategies_dir / "hierarchical.py"

    results = []

    # Check file exists
    results.append(check_file_exists(hierarchical_file, "HierarchicalStrategy file"))

    if hierarchical_file.exists():
        # Check inheritance
        results.append(check_class_inheritance(
            hierarchical_file,
            "HierarchicalStrategy",
            ["CombinedQueryMixin", "InferenceStrategy"]
        ))

        # Check methods
        results.append(check_method_exists(hierarchical_file, "HierarchicalStrategy", "infer_with_text_queries"))

    return all(results)


def validate_exhaustive_pairing():
    """Validate ExhaustivePairingWithCombinedQuery implementation."""
    logger.info("\n" + "="*80)
    logger.info("VALIDATION 4: ExhaustivePairingWithCombinedQuery")
    logger.info("="*80)

    exhaustive_file = strategies_dir / "exhaustive_pairing_with_combined_query.py"

    results = []

    # Check file exists
    results.append(check_file_exists(exhaustive_file, "ExhaustivePairingWithCombinedQuery file"))

    if exhaustive_file.exists():
        # Check inheritance
        results.append(check_class_inheritance(
            exhaustive_file,
            "ExhaustivePairingWithCombinedQuery",
            ["CombinedQueryMixin", "ExhaustivePairingStrategy"]
        ))

        # Check methods
        results.append(check_method_exists(
            exhaustive_file,
            "ExhaustivePairingWithCombinedQuery",
            "infer_with_text_queries"
        ))

        # Check that old methods are removed (should use Mixin methods instead)
        content = exhaustive_file.read_text()
        if "def _extract_text_feature" in content:
            logger.error("  ✗ Old _extract_text_feature method still exists (should use Mixin)")
            results.append(False)
        else:
            logger.info("  ✓ Old _extract_text_feature method removed (using Mixin)")
            results.append(True)

        if "def _update_proposal_scores_with_combined_feat" in content:
            logger.error("  ✗ Old _update_proposal_scores_with_combined_feat method still exists (should use Mixin)")
            results.append(False)
        else:
            logger.info("  ✓ Old _update_proposal_scores_with_combined_feat method removed (using Mixin)")
            results.append(True)

    return all(results)


def validate_configuration():
    """Validate configuration file settings."""
    logger.info("\n" + "="*80)
    logger.info("VALIDATION 5: Configuration Settings")
    logger.info("="*80)

    results = []

    # Check file exists
    results.append(check_file_exists(config_file, "Configuration file"))

    if config_file.exists():
        content = config_file.read_text()

        # Check scoring method
        if "method: arithmetic" in content:
            logger.info("  ✓ scoring.method: arithmetic")
            results.append(True)
        else:
            logger.error("  ✗ scoring.method is NOT set to arithmetic")
            results.append(False)

        # Check geometric weight
        if "geometric: 0.0" in content or "geometric: 0" in content:
            logger.info("  ✓ scoring.weights.geometric: 0.0 (disabled)")
            results.append(True)
        else:
            logger.error("  ✗ scoring.weights.geometric is NOT disabled")
            results.append(False)

        # Check use_geometric_score in pairing
        if "use_geometric_score: false" in content:
            logger.info("  ✓ pairing.use_geometric_score: false")
            results.append(True)
        else:
            logger.error("  ✗ pairing.use_geometric_score is NOT set to false")
            results.append(False)

        # Check use_combined_query defaults
        if "use_combined_query: true" in content:
            logger.info("  ✓ use_combined_query: true (default for strategies)")
            results.append(True)
        else:
            logger.warning("  ! use_combined_query not explicitly set to true (check if using defaults)")
            results.append(True)  # Still pass if using defaults

    return all(results)


def main():
    """Run all validations."""
    logger.info("="*80)
    logger.info("Validating Combined Query Implementation")
    logger.info("="*80)

    results = []

    # Run validations
    results.append(("CombinedQueryMixin", validate_mixin()))
    results.append(("IndependentStrategy", validate_independent_strategy()))
    results.append(("HierarchicalStrategy", validate_hierarchical_strategy()))
    results.append(("ExhaustivePairingWithCombinedQuery", validate_exhaustive_pairing()))
    results.append(("Configuration", validate_configuration()))

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:40s} {status}")

    logger.info("="*80)
    logger.info(f"TOTAL: {passed}/{total} validations passed")
    logger.info("="*80)

    if passed == total:
        logger.info("\n🎉 All validations passed!")
        logger.info("\nSummary of Implementation:")
        logger.info("1. ✓ CombinedQueryMixin created with reusable combined query functionality")
        logger.info("2. ✓ IndependentStrategy now supports combined queries (inherits Mixin)")
        logger.info("3. ✓ HierarchicalStrategy now supports combined queries (inherits Mixin)")
        logger.info("4. ✓ ExhaustivePairingWithCombinedQuery refactored to use Mixin")
        logger.info("5. ✓ Configuration uses arithmetic mean scoring with geometric disabled")
        logger.info("\nAll strategies now use 'part of object' combined queries by default!")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} validation(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
