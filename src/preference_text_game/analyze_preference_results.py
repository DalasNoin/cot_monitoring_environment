#!/usr/bin/env python3
"""
Analyze and visualize preference consistency results from inspect-ai experiments.
Shows stated vs revealed preferences breakdown by category pairs.
"""

import json
import zipfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from dataclasses import dataclass
import argparse


@dataclass
class PreferenceResult:
    """Single preference test result."""
    category_pair: str
    stated: str
    revealed: str
    is_consistent: bool
    model: str


def load_eval_results(eval_file: Path) -> List[PreferenceResult]:
    """Load and parse results from inspect eval file."""
    results = []

    with zipfile.ZipFile(eval_file, 'r') as z:
        # Get all sample files
        sample_files = [f for f in z.namelist() if f.startswith('samples/')]

        for sample_file in sample_files:
            with z.open(sample_file) as f:
                sample = json.load(f)

            # Extract scoring metadata
            scores = sample.get('scores', {})
            vector_score = scores.get('vector_consistency_scorer', {})
            metadata = vector_score.get('metadata', {})

            # Get preference details
            stated = metadata.get('stated_readable', 'Unknown')
            revealed = metadata.get('revealed_readable', 'Unknown')
            consistency = metadata.get('trial_consistency', 0)
            category_pair = metadata.get('category_pair', 'unknown')

            # Get model name from output
            output = sample.get('output', {})
            model = output.get('model', 'unknown')

            # Clean up category pair name
            if '_vs_' in category_pair:
                cat_a, cat_b = category_pair.split('_vs_')
                category_pair = f"{cat_a} vs {cat_b}"

            results.append(PreferenceResult(
                category_pair=category_pair,
                stated=stated,
                revealed=revealed,
                is_consistent=(consistency >= 0.99),  # Allow for floating point
                model=model
            ))

    return results


def create_preference_matrix(results: List[PreferenceResult]) -> pd.DataFrame:
    """Create matrix showing stated vs revealed preferences."""
    # Count occurrences
    matrix_data = defaultdict(lambda: defaultdict(int))

    for result in results:
        matrix_data[result.stated][result.revealed] += 1

    # Convert to DataFrame
    df = pd.DataFrame(matrix_data).fillna(0).astype(int)

    # Ensure square matrix with all categories
    all_categories = set()
    for result in results:
        all_categories.add(result.stated)
        all_categories.add(result.revealed)

    for cat in all_categories:
        if cat not in df.columns:
            df[cat] = 0
        if cat not in df.index:
            df.loc[cat] = 0

    # Sort for consistent ordering
    categories = sorted(all_categories)
    df = df.reindex(index=categories, columns=categories, fill_value=0)

    return df


def plot_preference_analysis(results: List[PreferenceResult], output_dir: Path):
    """Create comprehensive visualization of preferences."""

    # Set up the figure with cleaner layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # Get unique models and category pairs
    models = list(set(r.model for r in results))
    category_pairs = list(set(r.category_pair for r in results))

    # 1. Overall stated vs revealed matrix
    ax1 = fig.add_subplot(gs[0, 0])
    matrix_df = create_preference_matrix(results)
    sns.heatmap(matrix_df, annot=True, fmt='d', cmap='Blues', ax=ax1,
                cbar_kws={'label': 'Count'}, square=True)
    ax1.set_title('Stated vs Revealed Preferences', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('What They Actually Chose', fontsize=12)
    ax1.set_ylabel('What They Said They Prefer', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)

    # 2. Consistency by category pair
    ax2 = fig.add_subplot(gs[0, 1])
    pair_consistency = defaultdict(list)
    for result in results:
        pair_consistency[result.category_pair].append(1 if result.is_consistent else 0)

    pair_names = list(pair_consistency.keys())
    pair_rates = [np.mean(values) * 100 for values in pair_consistency.values()]

    # Clean up pair names for display
    clean_names = []
    for name in pair_names:
        # Shorten long category names
        name = name.replace('social/interpersonal work', 'social')
        name = name.replace('technical/systematic work', 'technical')
        name = name.replace('emotional support work', 'emotional support')
        name = name.replace('community building work', 'community')
        name = name.replace('teaching/mentoring work', 'teaching')
        name = name.replace('advocacy/persuasion work', 'advocacy')
        name = name.replace(' work', '')
        clean_names.append(name)

    y_pos = np.arange(len(clean_names))
    bars = ax2.barh(y_pos, pair_rates, color='steelblue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(clean_names, fontsize=11)
    ax2.set_xlabel('Consistency Rate (%)', fontsize=12)
    ax2.set_title('Consistency by Category Pair', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlim([0, 100])
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, pair_rates)):
        ax2.text(rate + 2, bar.get_y() + bar.get_height()/2, f'{rate:.0f}%',
                va='center', fontsize=10, fontweight='bold')

    # 3. Top inconsistency patterns
    ax3 = fig.add_subplot(gs[1, :])
    inconsistent = [r for r in results if not r.is_consistent]

    if inconsistent and len(inconsistent) >= 3:  # Only show if we have meaningful data
        # Create flow data
        flow_data = defaultdict(int)
        for r in inconsistent:
            # Clean up category names
            stated_clean = r.stated.replace(' work', '').replace('social/interpersonal', 'social').replace('technical/systematic', 'technical').replace('emotional support', 'emotional').replace('community building', 'community').replace('teaching/mentoring', 'teaching').replace('advocacy/persuasion', 'advocacy')
            revealed_clean = r.revealed.replace(' work', '').replace('social/interpersonal', 'social').replace('technical/systematic', 'technical').replace('emotional support', 'emotional').replace('community building', 'community').replace('teaching/mentoring', 'teaching').replace('advocacy/persuasion', 'advocacy')
            flow_data[(stated_clean, revealed_clean)] += 1

        # Sort by frequency and take top 8
        flows = sorted(flow_data.items(), key=lambda x: x[1], reverse=True)[:8]

        if flows:
            labels = [f"Said '{s}' → Chose '{r}'" for (s, r), _ in flows]
            values = [v for _, v in flows]

            y_pos = np.arange(len(labels))
            bars = ax3.barh(y_pos, values, color='coral', alpha=0.7)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(labels, fontsize=11)
            ax3.set_xlabel('Number of Cases', fontsize=12)
            ax3.set_title('Most Common Inconsistency Patterns', fontsize=14, fontweight='bold', pad=20)
            ax3.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax3.text(val + 0.2, bar.get_y() + bar.get_height()/2, str(val),
                        va='center', fontsize=10, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'High Consistency!\nFew discrepancies found.',
                transform=ax3.transAxes, ha='center', va='center',
                fontsize=16, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax3.set_title('Inconsistency Patterns', fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])

    # 4. Summary statistics (top right, cleaner)
    total = len(results)
    consistent = sum(1 for r in results if r.is_consistent)
    consistency_rate = (consistent / total * 100) if total > 0 else 0

    # Create summary text box in bottom right of figure
    summary_text = f"""SUMMARY

Total Tests: {total:,}
Consistent: {consistent:,} ({consistency_rate:.1f}%)
Inconsistent: {total - consistent:,} ({100 - consistency_rate:.1f}%)

Models: {len(models)}
Category Pairs: {len(category_pairs)}"""

    fig.text(0.02, 0.02, summary_text, fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    # Main title
    model_str = models[0] if len(models) == 1 else f"{len(models)} models"
    plt.suptitle(f'Preference Consistency Analysis - {model_str}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    output_file = output_dir / 'preference_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved visualization to {output_file}")

    return fig


def generate_category_report(results: List[PreferenceResult], output_dir: Path):
    """Generate detailed text report by category pair."""

    report = []
    report.append("=" * 80)
    report.append("PREFERENCE CONSISTENCY ANALYSIS - DETAILED REPORT")
    report.append("=" * 80)
    report.append("")

    # Group by category pair
    by_pair = defaultdict(list)
    for r in results:
        by_pair[r.category_pair].append(r)

    # Sort pairs by consistency rate
    pair_stats = []
    for pair, pair_results in by_pair.items():
        consistency_rate = sum(1 for r in pair_results if r.is_consistent) / len(pair_results)
        pair_stats.append((pair, pair_results, consistency_rate))

    pair_stats.sort(key=lambda x: x[2])  # Sort by consistency (lowest first)

    for pair, pair_results, consistency_rate in pair_stats:
        report.append(f"\n{pair.upper()}")
        report.append("-" * len(pair))
        report.append(f"Samples: {len(pair_results)}")
        report.append(f"Consistency: {consistency_rate:.1%}")
        report.append("")

        # Count preferences
        stated_counts = defaultdict(int)
        revealed_counts = defaultdict(int)

        for r in pair_results:
            stated_counts[r.stated] += 1
            revealed_counts[r.revealed] += 1

        report.append("Stated Preferences:")
        for pref, count in sorted(stated_counts.items(), key=lambda x: -x[1]):
            report.append(f"  - {pref}: {count} ({count/len(pair_results)*100:.0f}%)")

        report.append("\nRevealed Preferences:")
        for pref, count in sorted(revealed_counts.items(), key=lambda x: -x[1]):
            report.append(f"  - {pref}: {count} ({count/len(pair_results)*100:.0f}%)")

        # Show discrepancies
        inconsistent = [r for r in pair_results if not r.is_consistent]
        if inconsistent:
            report.append(f"\nDiscrepancies ({len(inconsistent)} cases):")
            discrepancy_counts = defaultdict(int)
            for r in inconsistent:
                discrepancy_counts[(r.stated, r.revealed)] += 1

            for (stated, revealed), count in sorted(discrepancy_counts.items(), key=lambda x: -x[1]):
                report.append(f"  - Said '{stated}' but chose '{revealed}': {count}")

        report.append("")

    # Write report
    report_file = output_dir / 'preference_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"📝 Saved detailed report to {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze preference consistency results')
    parser.add_argument('--log-dir', type=str, default='vector_results',
                       help='Directory containing inspect eval logs')
    parser.add_argument('--output-dir', type=str, default='preference_analysis',
                       help='Directory to save analysis outputs')

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find the latest eval file
    eval_files = list(log_dir.glob('*.eval'))
    if not eval_files:
        print(f"❌ No eval files found in {log_dir}")
        return

    latest_eval = max(eval_files, key=lambda f: f.stat().st_mtime)
    print(f"📂 Analyzing: {latest_eval.name}")

    # Load results
    results = load_eval_results(latest_eval)
    print(f"📊 Loaded {len(results)} preference test results")

    # Create visualizations
    plot_preference_analysis(results, output_dir)

    # Generate detailed report
    generate_category_report(results, output_dir)

    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    print(f"   - preference_analysis.png: Visual analysis")
    print(f"   - preference_report.txt: Detailed text report")


if __name__ == "__main__":
    main()