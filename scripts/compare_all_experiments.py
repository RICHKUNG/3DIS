#!/usr/bin/env python3
"""
Compare results from multiple SigLIP sanity check experiments.

This script loads results from different experimental conditions and
compares them side-by-side.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_result(path: Path) -> Dict:
    """Load a single result JSON."""
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)


def extract_metrics(result: Dict) -> Dict:
    """Extract key metrics from a result."""
    if result is None:
        return None

    return {
        'num_proposals': result.get('num_proposals', 0),
        'num_labels': result.get('num_labels', 0),
        'num_with_text': result.get('num_with_text', 0),
        'rank_1_pct': result.get('rank_1_percentage', 0.0),
        'mean_correct_sim': result.get('mean_correct_similarity'),
        'mean_margin': result.get('mean_margin'),
        'min_correct_sim': result.get('min_correct_similarity'),
        'max_correct_sim': result.get('max_correct_similarity'),
    }


def print_comparison_table(experiments: Dict[str, Dict]):
    """Print comparison table of all experiments."""
    print("\n" + "="*100)
    print("EXPERIMENTAL COMPARISON")
    print("="*100)

    # Header
    print(f"\n{'Experiment':<30} {'N':<5} {'Rank-1':<10} {'Mean Sim':<12} {'Mean Margin':<15} {'Min/Max Sim':<20}")
    print("-"*100)

    # Rows
    for exp_name, metrics in experiments.items():
        if metrics is None:
            print(f"{exp_name:<30} (NOT AVAILABLE)")
            continue

        n = metrics['num_with_text']
        rank1 = f"{metrics['rank_1_pct']:.1f}%"
        mean_sim = f"{metrics['mean_correct_sim']:.4f}" if metrics['mean_correct_sim'] is not None else "N/A"
        mean_margin = f"{metrics['mean_margin']:.4f}" if metrics['mean_margin'] is not None else "N/A"

        if metrics['min_correct_sim'] is not None and metrics['max_correct_sim'] is not None:
            min_max = f"[{metrics['min_correct_sim']:.4f}, {metrics['max_correct_sim']:.4f}]"
        else:
            min_max = "N/A"

        print(f"{exp_name:<30} {n:<5} {rank1:<10} {mean_sim:<12} {mean_margin:<15} {min_max:<20}")


def analyze_improvements(baseline: Dict, experiment: Dict, exp_name: str):
    """Analyze improvements compared to baseline."""
    if baseline is None or experiment is None:
        return

    print(f"\n{exp_name} vs Baseline:")

    # Rank-1 improvement
    rank1_diff = experiment['rank_1_pct'] - baseline['rank_1_pct']
    print(f"  Rank-1: {baseline['rank_1_pct']:.1f}% → {experiment['rank_1_pct']:.1f}% ({rank1_diff:+.1f}%)")

    # Mean similarity improvement
    if baseline['mean_correct_sim'] is not None and experiment['mean_correct_sim'] is not None:
        sim_diff = experiment['mean_correct_sim'] - baseline['mean_correct_sim']
        sim_pct = (sim_diff / baseline['mean_correct_sim']) * 100 if baseline['mean_correct_sim'] != 0 else 0
        print(f"  Mean similarity: {baseline['mean_correct_sim']:.4f} → {experiment['mean_correct_sim']:.4f} ({sim_pct:+.1f}%)")

    # Margin improvement
    if baseline['mean_margin'] is not None and experiment['mean_margin'] is not None:
        margin_diff = experiment['mean_margin'] - baseline['mean_margin']
        print(f"  Mean margin: {baseline['mean_margin']:.4f} → {experiment['mean_margin']:.4f} ({margin_diff:+.4f})")


def analyze_per_category(results_dict: Dict[str, Dict], category_name: str = "cabinet door"):
    """Analyze specific category across experiments."""
    print(f"\n{'='*100}")
    print(f"PER-CATEGORY ANALYSIS: '{category_name}'")
    print(f"{'='*100}")

    for exp_name, result in results_dict.items():
        if result is None:
            continue

        # Find instances of this category
        instances = [d for d in result.get('details', [])
                     if category_name in d.get('text', '').lower()]

        if not instances:
            print(f"\n{exp_name}: No instances found")
            continue

        # Statistics for this category
        sims = [inst['correct_similarity'] for inst in instances if inst['correct_similarity'] is not None]
        ranks = [inst['rank'] for inst in instances if inst['rank'] is not None]
        margins = [inst['margin'] for inst in instances if inst['margin'] is not None]

        if not sims:
            print(f"\n{exp_name}: No valid data")
            continue

        rank1_count = sum(1 for r in ranks if r == 1)

        print(f"\n{exp_name}:")
        print(f"  Instances: {len(instances)}")
        print(f"  Rank-1: {rank1_count}/{len(instances)} ({100*rank1_count/len(instances):.1f}%)")
        print(f"  Mean similarity: {np.mean(sims):.4f}")
        print(f"  Mean margin: {np.mean(margins):.4f}" if margins else "  Mean margin: N/A")
        print(f"  Similarity range: [{np.min(sims):.4f}, {np.max(sims):.4f}]")


def main():
    parser = argparse.ArgumentParser(description='Compare SigLIP sanity check experiments')
    parser.add_argument('--baseline', type=Path, required=True,
                        help='Baseline result JSON (e.g., topk=1, scale=300)')
    parser.add_argument('--topk5', type=Path,
                        help='Result with topk=5')
    parser.add_argument('--no-softmax', type=Path,
                        help='Result without softmax')
    parser.add_argument('--scale100', type=Path,
                        help='Result with scale=100')
    parser.add_argument('--original', type=Path,
                        help='Result from original check_siglip_gt_similarity.py')
    parser.add_argument('--category', type=str, default='cabinet door',
                        help='Category to analyze in detail')

    args = parser.parse_args()

    # Load all results
    print("Loading results...")
    results = {}
    experiments = {}

    # Baseline
    baseline_result = load_result(args.baseline)
    if baseline_result is None:
        print(f"Error: Baseline result not found at {args.baseline}")
        sys.exit(1)
    results['Baseline (topk=1, scale=300)'] = baseline_result
    experiments['Baseline (topk=1, scale=300)'] = extract_metrics(baseline_result)

    # Other experiments
    if args.topk5:
        results['TopK=5'] = load_result(args.topk5)
        experiments['TopK=5'] = extract_metrics(results['TopK=5'])

    if args.no_softmax:
        results['No Softmax'] = load_result(args.no_softmax)
        experiments['No Softmax'] = extract_metrics(results['No Softmax'])

    if args.scale100:
        results['Scale=100'] = load_result(args.scale100)
        experiments['Scale=100'] = extract_metrics(results['Scale=100'])

    if args.original:
        results['Original Script'] = load_result(args.original)
        experiments['Original Script'] = extract_metrics(results['Original Script'])

    # Print comparison table
    print_comparison_table(experiments)

    # Analyze improvements
    print(f"\n{'='*100}")
    print("IMPROVEMENT ANALYSIS")
    print(f"{'='*100}")

    baseline_metrics = experiments['Baseline (topk=1, scale=300)']

    for exp_name, metrics in experiments.items():
        if exp_name == 'Baseline (topk=1, scale=300)':
            continue
        analyze_improvements(baseline_metrics, metrics, exp_name)

    # Per-category analysis
    analyze_per_category(results, args.category)

    # Best configuration
    print(f"\n{'='*100}")
    print("BEST CONFIGURATION")
    print(f"{'='*100}")

    valid_experiments = {k: v for k, v in experiments.items() if v is not None}
    if valid_experiments:
        best_rank1 = max(valid_experiments.items(), key=lambda x: x[1]['rank_1_pct'])
        best_margin = max(valid_experiments.items(), key=lambda x: x[1]['mean_margin'] or float('-inf'))

        print(f"\nBest Rank-1 accuracy: {best_rank1[0]} ({best_rank1[1]['rank_1_pct']:.1f}%)")
        print(f"Best mean margin: {best_margin[0]} ({best_margin[1]['mean_margin']:.4f})")


if __name__ == '__main__':
    main()
