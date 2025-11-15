#!/usr/bin/env python3
"""
Diagnose SigLIP similarity issues by examining the similarity matrix in detail.

This script loads results from siglip_gt_sanity_check_with_scaling.py and
performs detailed analysis of the similarity scores.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_results(json_path: Path) -> dict:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_similarity_distribution(results: dict):
    """Analyze the distribution of similarity scores."""
    print("\n" + "="*80)
    print("SIMILARITY DISTRIBUTION ANALYSIS")
    print("="*80)

    details = results['details']

    # Extract similarity scores
    correct_sims = []
    incorrect_sims = []

    for detail in details:
        if detail['correct_similarity'] is not None:
            correct_sims.append(detail['correct_similarity'])
            if detail['max_incorrect_similarity'] is not None:
                incorrect_sims.append(detail['max_incorrect_similarity'])

    if not correct_sims:
        print("No valid similarity scores found!")
        return

    print(f"\nCorrect Matches (GT label):")
    print(f"  Count: {len(correct_sims)}")
    print(f"  Mean: {np.mean(correct_sims):.4f}")
    print(f"  Std: {np.std(correct_sims):.4f}")
    print(f"  Min: {np.min(correct_sims):.4f}")
    print(f"  Max: {np.max(correct_sims):.4f}")
    print(f"  Median: {np.median(correct_sims):.4f}")
    print(f"  Percentiles:")
    for p in [25, 50, 75, 90, 95]:
        print(f"    {p}th: {np.percentile(correct_sims, p):.4f}")

    if incorrect_sims:
        print(f"\nIncorrect Matches (Max wrong label):")
        print(f"  Count: {len(incorrect_sims)}")
        print(f"  Mean: {np.mean(incorrect_sims):.4f}")
        print(f"  Std: {np.std(incorrect_sims):.4f}")
        print(f"  Min: {np.min(incorrect_sims):.4f}")
        print(f"  Max: {np.max(incorrect_sims):.4f}")
        print(f"  Median: {np.median(incorrect_sims):.4f}")

    # Margin distribution
    margins = [d['margin'] for d in details if d['margin'] is not None]
    if margins:
        print(f"\nMargin (Correct - Max Incorrect):")
        print(f"  Mean: {np.mean(margins):.4f}")
        print(f"  Std: {np.std(margins):.4f}")
        print(f"  Min: {np.min(margins):.4f}")
        print(f"  Max: {np.max(margins):.4f}")
        print(f"  Positive margins: {sum(1 for m in margins if m > 0)}/{len(margins)} ({100*sum(1 for m in margins if m > 0)/len(margins):.1f}%)")


def analyze_by_category(results: dict):
    """Analyze results grouped by category."""
    print("\n" + "="*80)
    print("CATEGORY-WISE ANALYSIS")
    print("="*80)

    # Group by label
    label_groups = {}
    for detail in results['details']:
        if detail['correct_similarity'] is None:
            continue

        label = detail['text']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(detail)

    print(f"\nFound {len(label_groups)} unique labels:")

    for label, instances in sorted(label_groups.items()):
        sims = [inst['correct_similarity'] for inst in instances]
        ranks = [inst['rank'] for inst in instances]
        rank1_count = sum(1 for r in ranks if r == 1)

        print(f"\n  '{label}': {len(instances)} instances")
        print(f"    Rank-1: {rank1_count}/{len(instances)} ({100*rank1_count/len(instances):.1f}%)")
        print(f"    Mean similarity: {np.mean(sims):.4f}")
        print(f"    Similarity range: [{np.min(sims):.4f}, {np.max(sims):.4f}]")


def show_confusion_patterns(results: dict):
    """Show which labels are commonly confused with each other."""
    print("\n" + "="*80)
    print("CONFUSION PATTERNS")
    print("="*80)

    # For simplicity, we don't have the full similarity matrix here
    # Just show the worst cases

    worst_cases = [d for d in results['details']
                   if d['margin'] is not None and d['margin'] < 0]
    worst_cases.sort(key=lambda x: x['margin'])

    print(f"\nWorst {min(10, len(worst_cases))} cases (lowest margin):")
    for i, case in enumerate(worst_cases[:10], 1):
        print(f"\n  {i}. GT: '{case['text']}' (ID {case['raw_label']})")
        print(f"     Correct similarity: {case['correct_similarity']:.4f}")
        print(f"     Max incorrect: {case['max_incorrect_similarity']:.4f}")
        print(f"     Margin: {case['margin']:.4f}")
        print(f"     Rank: {case['rank']}")


def plot_similarity_histogram(results: dict, output_path: Path = None):
    """Plot histogram of similarity scores."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return

    details = results['details']

    correct_sims = [d['correct_similarity'] for d in details if d['correct_similarity'] is not None]
    incorrect_sims = [d['max_incorrect_similarity'] for d in details if d['max_incorrect_similarity'] is not None]

    if not correct_sims:
        print("No data to plot!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(correct_sims, bins=20, alpha=0.7, label='Correct', color='green')
    if incorrect_sims:
        axes[0].hist(incorrect_sims, bins=20, alpha=0.7, label='Max Incorrect', color='red')
    axes[0].set_xlabel('Similarity Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Similarity Score Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    data_to_plot = []
    labels_to_plot = []
    if correct_sims:
        data_to_plot.append(correct_sims)
        labels_to_plot.append('Correct')
    if incorrect_sims:
        data_to_plot.append(incorrect_sims)
        labels_to_plot.append('Max Incorrect')

    axes[1].boxplot(data_to_plot, labels=labels_to_plot)
    axes[1].set_ylabel('Similarity Score')
    axes[1].set_title('Similarity Score Box Plot')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def compare_with_without_softmax(with_softmax_path: Path, without_softmax_path: Path):
    """Compare results with and without softmax."""
    print("\n" + "="*80)
    print("SOFTMAX COMPARISON")
    print("="*80)

    with_sm = load_results(with_softmax_path)
    without_sm = load_results(without_softmax_path)

    print(f"\nWith Softmax:")
    print(f"  Rank-1 accuracy: {with_sm['rank_1_percentage']:.1f}%")
    print(f"  Mean correct similarity: {with_sm['mean_correct_similarity']:.4f}")
    print(f"  Mean margin: {with_sm['mean_margin']:.4f}")

    print(f"\nWithout Softmax:")
    print(f"  Rank-1 accuracy: {without_sm['rank_1_percentage']:.1f}%")
    print(f"  Mean correct similarity: {without_sm['mean_correct_similarity']:.4f}")
    print(f"  Mean margin: {without_sm['mean_margin']:.4f}")

    # Check if instances are in the same order
    if len(with_sm['details']) == len(without_sm['details']):
        print(f"\nPer-instance comparison:")

        improved = 0
        degraded = 0
        unchanged = 0

        for ws, wos in zip(with_sm['details'], without_sm['details']):
            if ws['margin'] is None or wos['margin'] is None:
                continue

            if ws['margin'] > wos['margin']:
                improved += 1
            elif ws['margin'] < wos['margin']:
                degraded += 1
            else:
                unchanged += 1

        total = improved + degraded + unchanged
        if total > 0:
            print(f"  Improved with softmax: {improved}/{total} ({100*improved/total:.1f}%)")
            print(f"  Degraded with softmax: {degraded}/{total} ({100*degraded/total:.1f}%)")
            print(f"  Unchanged: {unchanged}/{total} ({100*unchanged/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Diagnose SigLIP similarity issues')
    parser.add_argument('--results', type=Path, required=True, help='Path to results JSON file')
    parser.add_argument('--compare-with', type=Path, help='Path to another results JSON to compare')
    parser.add_argument('--plot', type=Path, help='Output path for plots')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')

    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: Results file not found: {args.results}")
        sys.exit(1)

    print(f"Loading results from: {args.results}")
    results = load_results(args.results)

    # Basic stats
    print(f"\nDataset: {results['num_proposals']} proposals, {results['num_labels']} labels")
    print(f"Rank-1 accuracy: {results['rank_1_percentage']:.1f}%")

    # Detailed analysis
    analyze_similarity_distribution(results)
    analyze_by_category(results)
    show_confusion_patterns(results)

    # Comparison
    if args.compare_with:
        if not args.compare_with.exists():
            print(f"\nWarning: Comparison file not found: {args.compare_with}")
        else:
            compare_with_without_softmax(args.results, args.compare_with)

    # Plotting
    if not args.no_plot:
        try:
            plot_similarity_histogram(results, args.plot)
        except Exception as e:
            print(f"\nWarning: Could not create plots: {e}")


if __name__ == '__main__':
    main()
