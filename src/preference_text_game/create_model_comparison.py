#!/usr/bin/env python3
"""
Create intuitive model comparison plots for preference consistency analysis.
Shows how different models behave across all task categories.
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
class ModelResult:
    """Results for one model across all tasks."""
    model: str
    category_preferences: Dict[str, Dict[str, int]]  # category -> {stated/revealed -> count}
    consistency_by_pair: Dict[str, float]  # category_pair -> consistency_rate
    total_tests: int


def load_all_model_results(eval_file: Path) -> List[ModelResult]:
    """Load and organize results by model."""

    # Raw results by model
    model_data = defaultdict(list)

    with zipfile.ZipFile(eval_file, 'r') as z:
        sample_files = [f for f in z.namelist() if f.startswith('samples/')]

        for sample_file in sample_files:
            with z.open(sample_file) as f:
                sample = json.load(f)

            # Extract data
            scores = sample.get('scores', {})
            vector_score = scores.get('vector_consistency_scorer', {})
            metadata = vector_score.get('metadata', {})

            output = sample.get('output', {})
            model = output.get('model', 'unknown')

            # Clean model name
            if 'openrouter/' in model:
                model = model.replace('openrouter/', '').replace('openai/', '').replace('anthropic/', '')

            result = {
                'model': model,
                'stated': metadata.get('stated_readable', 'Unknown'),
                'revealed': metadata.get('revealed_readable', 'Unknown'),
                'consistency': metadata.get('trial_consistency', 0),
                'category_pair': metadata.get('category_pair', 'unknown')
            }

            model_data[model].append(result)

    # Process each model
    model_results = []
    for model, results in model_data.items():
        # Aggregate category preferences
        category_prefs = defaultdict(lambda: {'stated': 0, 'revealed': 0})
        consistency_by_pair = defaultdict(list)

        for r in results:
            # Count stated preferences
            stated_cat = clean_category_name(r['stated'])
            revealed_cat = clean_category_name(r['revealed'])

            if stated_cat != 'unknown':
                category_prefs[stated_cat]['stated'] += 1
            if revealed_cat != 'unknown':
                category_prefs[revealed_cat]['revealed'] += 1

            # Track consistency by pair
            pair = r['category_pair'].replace('_vs_', ' vs ')
            consistency_by_pair[pair].append(r['consistency'])

        # Calculate average consistency by pair
        pair_consistency = {
            pair: np.mean(values) for pair, values in consistency_by_pair.items()
        }

        model_results.append(ModelResult(
            model=model,
            category_preferences=dict(category_prefs),
            consistency_by_pair=pair_consistency,
            total_tests=len(results)
        ))

    return model_results


def clean_category_name(category: str) -> str:
    """Clean and shorten category names."""
    if not category or category == 'Unknown':
        return 'unknown'

    category = category.lower()

    # Map to shorter names
    if 'creative' in category:
        return 'Creative'
    elif 'repetitive' in category:
        return 'Repetitive'
    elif 'analytical' in category:
        return 'Analytical'
    elif 'social' in category or 'interpersonal' in category:
        return 'Social'
    elif 'technical' in category or 'systematic' in category:
        return 'Technical'
    elif 'emotional' in category or 'support' in category:
        return 'Emotional Support'
    elif 'community' in category or 'building' in category:
        return 'Community'
    elif 'teaching' in category or 'mentoring' in category:
        return 'Teaching'
    elif 'advocacy' in category or 'persuasion' in category:
        return 'Advocacy'
    elif 'preference' in category or 'neither' in category or 'quit' in category or 'refuse' in category:
        return 'No Preference'
    else:
        return 'Other'


def create_model_comparison_plot(model_results: List[ModelResult], output_dir: Path):
    """Create comprehensive model comparison visualization."""

    # Set up the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('AI Model Preference Analysis - Multi-Model Comparison', fontsize=16, fontweight='bold')

    models = [mr.model for mr in model_results]
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    # 1. Overall consistency comparison
    model_names = []
    overall_consistency = []

    for mr in model_results:
        model_names.append(mr.model)
        # Calculate overall consistency
        all_consistencies = []
        for consistencies in mr.consistency_by_pair.values():
            all_consistencies.append(consistencies)
        overall_consistency.append(np.mean(all_consistencies) * 100 if all_consistencies else 0)

    bars = ax1.bar(model_names, overall_consistency, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Overall Consistency (%)', fontsize=12)
    ax1.set_title('Model Consistency Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, overall_consistency):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.0f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Preference distribution by model (what they actually chose)
    all_categories = set()
    for mr in model_results:
        all_categories.update(mr.category_preferences.keys())
    all_categories = sorted([cat for cat in all_categories if cat != 'unknown'])

    # Create matrix for revealed preferences
    revealed_matrix = []
    for mr in model_results:
        row = []
        total_revealed = sum(prefs['revealed'] for prefs in mr.category_preferences.values())
        for cat in all_categories:
            count = mr.category_preferences.get(cat, {}).get('revealed', 0)
            percentage = (count / total_revealed * 100) if total_revealed > 0 else 0
            row.append(percentage)
        revealed_matrix.append(row)

    # Plot heatmap
    im = ax2.imshow(revealed_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(all_categories)))
    ax2.set_yticks(range(len(models)))
    ax2.set_xticklabels(all_categories, rotation=45, ha='right')
    ax2.set_yticklabels(models)
    ax2.set_title('What Models Actually Choose (%)', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(all_categories)):
            text = ax2.text(j, i, f'{revealed_matrix[i][j]:.0f}%',
                          ha="center", va="center", color="black" if revealed_matrix[i][j] < 50 else "white",
                          fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Percentage of Choices', rotation=270, labelpad=20)

    # 3. Stated vs Revealed comparison
    categories_for_comparison = ['Creative', 'Repetitive', 'Social', 'Technical', 'Analytical', 'Emotional Support', 'Community', 'Teaching', 'Advocacy']
    categories_for_comparison = [cat for cat in categories_for_comparison if cat in all_categories]

    x = np.arange(len(categories_for_comparison))
    width = 0.35

    # Calculate averages across all models
    avg_stated = []
    avg_revealed = []

    for cat in categories_for_comparison:
        stated_counts = []
        revealed_counts = []
        for mr in model_results:
            total_stated = sum(prefs['stated'] for prefs in mr.category_preferences.values())
            total_revealed = sum(prefs['revealed'] for prefs in mr.category_preferences.values())

            stated_pct = (mr.category_preferences.get(cat, {}).get('stated', 0) / total_stated * 100) if total_stated > 0 else 0
            revealed_pct = (mr.category_preferences.get(cat, {}).get('revealed', 0) / total_revealed * 100) if total_revealed > 0 else 0

            stated_counts.append(stated_pct)
            revealed_counts.append(revealed_pct)

        avg_stated.append(np.mean(stated_counts))
        avg_revealed.append(np.mean(revealed_counts))

    bars1 = ax3.bar(x - width/2, avg_stated, width, label='What They Say', color='lightblue', alpha=0.8, edgecolor='navy')
    bars2 = ax3.bar(x + width/2, avg_revealed, width, label='What They Do', color='orange', alpha=0.8, edgecolor='darkorange')

    ax3.set_xlabel('Task Category', fontsize=12)
    ax3.set_ylabel('Average Percentage', fontsize=12)
    ax3.set_title('Words vs Actions (Average Across Models)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories_for_comparison, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=9)

    # 4. Consistency by task pair comparison
    all_pairs = set()
    for mr in model_results:
        all_pairs.update(mr.consistency_by_pair.keys())
    all_pairs = sorted(all_pairs)

    # Plot consistency by pair for each model
    x_pos = np.arange(len(all_pairs))
    bar_width = 0.8 / len(models)

    for i, mr in enumerate(model_results):
        values = []
        for pair in all_pairs:
            consistency = mr.consistency_by_pair.get(pair, 0) * 100
            values.append(consistency)

        ax4.bar(x_pos + i * bar_width, values, bar_width,
               label=mr.model, color=colors[i], alpha=0.8)

    ax4.set_xlabel('Task Category Pairs', fontsize=12)
    ax4.set_ylabel('Consistency Rate (%)', fontsize=12)
    ax4.set_title('Consistency by Task Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos + bar_width * (len(models) - 1) / 2)
    ax4.set_xticklabels([pair.replace(' vs ', '\nvs\n') for pair in all_pairs],
                       rotation=0, ha='center', fontsize=9)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim([0, 100])

    plt.tight_layout()

    # Save
    output_file = output_dir / 'model_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved model comparison to {output_file}")

    return fig


def create_summary_report(model_results: List[ModelResult], output_dir: Path):
    """Create a summary report comparing models."""

    report = []
    report.append("=" * 80)
    report.append("MULTI-MODEL PREFERENCE CONSISTENCY COMPARISON")
    report.append("=" * 80)
    report.append("")

    # Overall statistics
    report.append("OVERALL RESULTS:")
    report.append("-" * 20)

    for mr in model_results:
        all_consistencies = []
        for consistencies in mr.consistency_by_pair.values():
            all_consistencies.append(consistencies)

        overall_consistency = np.mean(all_consistencies) if all_consistencies else 0

        report.append(f"{mr.model:15} | {mr.total_tests:3d} tests | {overall_consistency*100:5.1f}% consistent")

    report.append("")
    report.append("INTERPRETATION:")
    report.append("- Higher consistency = model's words match their actions")
    report.append("- Lower consistency = model says one thing but does another")
    report.append("")

    # Category preferences comparison
    report.append("WHAT MODELS ACTUALLY CHOOSE (Revealed Preferences):")
    report.append("-" * 50)

    all_categories = set()
    for mr in model_results:
        all_categories.update(mr.category_preferences.keys())
    all_categories = sorted([cat for cat in all_categories if cat != 'unknown'])

    # Header
    header = "Model".ljust(15)
    for cat in all_categories:
        header += f"{cat:>12}"
    report.append(header)
    report.append("-" * len(header))

    # Data rows
    for mr in model_results:
        row = mr.model.ljust(15)
        total_revealed = sum(prefs['revealed'] for prefs in mr.category_preferences.values())

        for cat in all_categories:
            count = mr.category_preferences.get(cat, {}).get('revealed', 0)
            percentage = (count / total_revealed * 100) if total_revealed > 0 else 0
            row += f"{percentage:11.1f}%"

        report.append(row)

    # Save report
    report_file = output_dir / 'model_comparison_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"📝 Saved comparison report to {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Create model comparison analysis')
    parser.add_argument('--log-dir', type=str, default='results/logs',
                       help='Directory containing inspect eval logs')
    parser.add_argument('--output-dir', type=str, default='results/model_comparison',
                       help='Directory to save comparison outputs')

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
    model_results = load_all_model_results(latest_eval)

    if not model_results:
        print("❌ No model results found")
        return

    print(f"📊 Found results for {len(model_results)} model(s):")
    for mr in model_results:
        print(f"  - {mr.model}: {mr.total_tests} tests")

    # Create visualizations
    create_model_comparison_plot(model_results, output_dir)
    create_summary_report(model_results, output_dir)

    print(f"\n✅ Model comparison complete! Results saved to {output_dir}/")
    print(f"   - model_comparison.png: Visual comparison")
    print(f"   - model_comparison_report.txt: Detailed comparison")


if __name__ == "__main__":
    main()