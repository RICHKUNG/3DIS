#!/usr/bin/env python3
"""
Verification script to check that all query similarity calculations
are using the correct method (scale_semantic_score + softmax).

This script verifies:
1. compute_cosine_similarity() uses scale + softmax
2. MultiLevelRetriever has scale_semantic_score parameter
3. All hierarchical.py calls pass scale_semantic_score
4. combined_query_mixin.py uses compute_cosine_similarity
5. assignment_utils.py already uses correct method
"""

import sys
from pathlib import Path
import re

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def check_file_contains(file_path: Path, patterns: list, description: str) -> bool:
    """Check if file contains all patterns."""
    try:
        content = file_path.read_text()

        missing = []
        for pattern in patterns:
            if not re.search(pattern, content, re.MULTILINE):
                missing.append(pattern)

        if missing:
            print(f"{RED}✗{RESET} {description}")
            for p in missing:
                print(f"  Missing pattern: {p}")
            return False
        else:
            print(f"{GREEN}✓{RESET} {description}")
            return True
    except Exception as e:
        print(f"{RED}✗{RESET} {description} - Error: {e}")
        return False


def main():
    root = Path(__file__).parent.parent

    print("="*80)
    print("VERIFYING QUERY SIMILARITY FIX")
    print("="*80)
    print()

    checks = []

    # Check 1: compute_cosine_similarity uses scale + softmax
    print("Check 1: compute_cosine_similarity() implementation")
    checks.append(check_file_contains(
        root / "src/my3dis/inference/retrieval.py",
        [
            r"scale_semantic_score.*=.*300\.0",
            r"scaled_similarities.*=.*scale_semantic_score.*\*.*similarities",
            r"np\.exp\(scaled_similarities",
            r"softmax_scores.*=.*exp_scores.*/"
        ],
        "  retrieval.py: compute_cosine_similarity() uses scale + softmax"
    ))

    # Check 2: MultiLevelRetriever has scale_semantic_score
    print("\nCheck 2: MultiLevelRetriever parameters")
    checks.append(check_file_contains(
        root / "src/my3dis/inference/retrieval.py",
        [
            r"scale_semantic_score:\s*float\s*=\s*300\.0",
            r"self\.scale_semantic_score\s*=\s*scale_semantic_score"
        ],
        "  retrieval.py: MultiLevelRetriever has scale_semantic_score parameter"
    ))

    # Check 3: MultiLevelRetriever.retrieve_single_level passes scale_semantic_score
    print("\nCheck 3: MultiLevelRetriever.retrieve_single_level()")
    checks.append(check_file_contains(
        root / "src/my3dis/inference/retrieval.py",
        [
            r"compute_cosine_similarity\(",
            r"scale_semantic_score=self\.scale_semantic_score"
        ],
        "  retrieval.py: retrieve_single_level() passes scale_semantic_score"
    ))

    # Check 4: hierarchical.py passes scale_semantic_score (4 places)
    print("\nCheck 4: hierarchical.py compute_cosine_similarity calls")
    hierarchical_content = (root / "src/my3dis/inference/strategies/hierarchical.py").read_text()
    count = len(re.findall(r"scale_semantic_score=self\.retriever\.scale_semantic_score", hierarchical_content))

    if count >= 4:
        print(f"{GREEN}✓{RESET}   hierarchical.py: All {count} calls pass scale_semantic_score")
        checks.append(True)
    else:
        print(f"{RED}✗{RESET}   hierarchical.py: Only {count}/4 calls pass scale_semantic_score")
        checks.append(False)

    # Check 5: combined_query_mixin.py uses compute_cosine_similarity
    print("\nCheck 5: combined_query_mixin.py update method")
    checks.append(check_file_contains(
        root / "src/my3dis/inference/strategies/combined_query_mixin.py",
        [
            r"from.*\.\.retrieval.*import.*compute_cosine_similarity",
            r"scores.*=.*compute_cosine_similarity\(",
            r"scale_semantic_score.*=.*scale_semantic_score"
        ],
        "  combined_query_mixin.py: update_proposals uses compute_cosine_similarity"
    ))

    # Check 6: assignment_utils.py already correct (just verify)
    print("\nCheck 6: assignment_utils.py (already correct)")
    checks.append(check_file_contains(
        root / "src/my3dis/siglip_assignment/assignment_utils.py",
        [
            r"scale_factor.*\*.*\(.*@.*text_features\.T.*\)",
            r"F\.softmax\(.*similarity"
        ],
        "  assignment_utils.py: Uses scale_factor * similarity + softmax"
    ))

    # Summary
    print("\n" + "="*80)
    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"{GREEN}✓ ALL CHECKS PASSED ({passed}/{total}){RESET}")
        print("\nAll query similarity calculations are using the correct method!")
        return 0
    else:
        print(f"{RED}✗ SOME CHECKS FAILED ({passed}/{total}){RESET}")
        print("\nPlease review the failed checks above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
