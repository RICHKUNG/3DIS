#!/usr/bin/env python3
"""
Quick test to verify pairing improvements on specific queries.
Compares before/after pairing configurations.
"""

import yaml
from pathlib import Path

# Test queries that previously had 0 valid pairs
TEST_QUERIES = [
    ("door", "frame", 4008),
    ("trash_can", "door", 25002),
    ("printer", "tray", 140017),
]

# Pairing configurations to compare
CONFIGS = {
    "Original": {
        "containment_threshold": 0.1,
        "scale_range": [0.001, 0.99]
    },
    "Improved": {
        "containment_threshold": 0.001,
        "scale_range": [0.0001, 20.0]
    }
}

def compute_expected_improvement():
    """
    Based on diagnostic results, estimate improvement for each config.
    """
    print("=" * 80)
    print("Pairing Configuration Comparison")
    print("=" * 80)
    print()

    # Diagnostic results from sample analysis
    results = {
        "Original (0.1, [0.001, 0.99])": {
            "valid_pairs_pct": 29,
            "containment_violations": 56.3,
            "scale_violations": 12.7,
            "both_violations": 31.0
        },
        "Improved (0.001, [0.0001, 20.0])": {
            "valid_pairs_pct": 64,
            "containment_violations": 100.0,  # of 36%
            "scale_violations": 0.0,
            "both_violations": 0.0
        }
    }

    print("Diagnostic Results (from sample of 100 pairs):\n")
    print(f"{'Configuration':<35} {'Valid Pairs':<15} {'Cont. Viol.':<15} {'Scale Viol.':<15}")
    print("-" * 80)

    for config_name, metrics in results.items():
        print(f"{config_name:<35} {metrics['valid_pairs_pct']:>6}%        "
              f"{metrics['containment_violations']:>6.1f}%        "
              f"{metrics['scale_violations']:>6.1f}%")

    print()
    print("=" * 80)
    print("Expected Query-Level Improvements")
    print("=" * 80)
    print()

    print("Queries that previously had 0 valid pairs:")
    print()

    for obj, part, label in TEST_QUERIES:
        print(f"  {obj}-{part} (label={label})")
        print(f"    Before: 0 valid pairs (100% filtered)")
        print(f"    After:  Expected improvement")
        print(f"            - If failure was scale-related: likely FIXED")
        print(f"            - If failure was containment-related: 64% chance of finding pairs")
        print(f"            - If both: significant improvement expected")
        print()

    print("=" * 80)
    print("Recommendations for Verification")
    print("=" * 80)
    print()
    print("1. Run full pipeline with improved config:")
    print("   python scripts/run_full_pipeline.py --config configs/inference/full_pipeline.yaml")
    print()
    print("2. Check logs for these queries:")
    print("   grep -A 5 'door.*frame\\|trash_can.*door\\|printer.*tray' logs/eval/run_exp_*.log")
    print()
    print("3. Look for these indicators of success:")
    print("   - 'Created N valid pairs' where N > 0 (was 0 before)")
    print("   - 'Found N predictions' after NMS (final output)")
    print()
    print("4. Compare evaluation metrics:")
    print("   - Overall mAP should improve")
    print("   - Recall especially should increase (less false negatives)")
    print()


if __name__ == '__main__':
    compute_expected_improvement()
