#!/usr/bin/env python3
"""
Create preference plots from JSON summary data instead of reading eval files.
This is faster and doesn't require the original eval files.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelData:
    """Model preference data loaded from JSON."""
    model: str
    total_tests: int
    stated_counts: Dict[str, int]
    revealed_counts: Dict[str, int]
    stated_percentages: Dict[str, float]
    revealed_percentages: Dict[str, float]


def load_json_data(json_file: Path) -> tuple[List[ModelData], Dict]:
    """Load model data and overall stats from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    models = []
    for model_name, model_data in data["models"].items():
        models.append(ModelData(
            model=model_name,
            total_tests=model_data["total_tests"],
            stated_counts=model_data["stated_preferences"],
            revealed_counts=model_data["revealed_preferences"],
            stated_percentages=model_data["stated_percentages"],
            revealed_percentages=model_data["revealed_percentages"]
        ))

    return models, data["overall_stats"]


def get_color(category: str, category_list: List[str]) -> str:
    """Get consistent color for a category."""
    if 'Refused' in category or 'Neither' in category:
        return '#FF0000'  # Bright red for refusals
    else:
        default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
            '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7'
        ]
        try:
            idx = sorted(category_list).index(category)
            return default_colors[idx % len(default_colors)]
        except (ValueError, IndexError):
            return '#808080'


def create_individual_plots(models: List[ModelData], categories: List[str], output_dir: Path):
    """Create individual plots for each model."""

    for model in models:
        # Create figure for this model (stated and revealed side by side)
        fig, (ax_stated, ax_revealed) = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f'{model.model}: Stated vs Revealed Preferences', fontsize=16, fontweight='bold')

        # Stated preferences pie chart
        stated_values = []
        stated_labels = []
        stated_colors = []

        for category in categories:
            count = model.stated_counts.get(category, 0)
            if count > 0:
                display_count = max(count, 0.01)
                stated_values.append(display_count)
                stated_labels.append(category)
                stated_colors.append(get_color(category, categories))

        def autopct_func_stated(pct):
            cumulative = 0
            for idx, val in enumerate(stated_values):
                cumulative += val / sum(stated_values) * 100
                if cumulative >= pct:
                    category = stated_labels[idx]
                    real_pct = model.stated_percentages.get(category, 0)
                    if real_pct < 0.5:
                        return f'{real_pct:.1f}%' if real_pct > 0 else '0%'
                    else:
                        return f'{real_pct:.0f}%'
            return f'{pct:.0f}%'

        if stated_values and sum(stated_values) > 0:
            wedges, texts, autotexts = ax_stated.pie(stated_values, labels=stated_labels,
                                                    colors=stated_colors, autopct=autopct_func_stated,
                                                    startangle=90)
            for autotext in autotexts:
                if float(autotext.get_text().rstrip('%')) < 5:
                    autotext.set_fontsize(9)
                else:
                    autotext.set_fontsize(11)
        else:
            ax_stated.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14,
                          transform=ax_stated.transAxes)

        ax_stated.set_title(f'What They SAY\n(Stated)', fontsize=14, fontweight='bold')

        # Revealed preferences pie chart
        revealed_values = []
        revealed_labels = []
        revealed_colors = []

        for category in categories:
            count = model.revealed_counts.get(category, 0)
            if count > 0:
                display_count = max(count, 0.01)
                revealed_values.append(display_count)
                revealed_labels.append(category)
                revealed_colors.append(get_color(category, categories))

        def autopct_func_revealed(pct):
            cumulative = 0
            for idx, val in enumerate(revealed_values):
                cumulative += val / sum(revealed_values) * 100
                if cumulative >= pct:
                    category = revealed_labels[idx]
                    real_pct = model.revealed_percentages.get(category, 0)
                    if real_pct < 0.5:
                        return f'{real_pct:.1f}%' if real_pct > 0 else '0%'
                    else:
                        return f'{real_pct:.0f}%'
            return f'{pct:.0f}%'

        if revealed_values and sum(revealed_values) > 0:
            wedges, texts, autotexts = ax_revealed.pie(revealed_values, labels=revealed_labels,
                                                      colors=revealed_colors, autopct=autopct_func_revealed,
                                                      startangle=90)
            for autotext in autotexts:
                if float(autotext.get_text().rstrip('%')) < 5:
                    autotext.set_fontsize(9)
                else:
                    autotext.set_fontsize(11)
        else:
            ax_revealed.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14,
                           transform=ax_revealed.transAxes)

        ax_revealed.set_title(f'What They DO\n(Revealed)', fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save individual model plot
        clean_model_name = model.model.replace('/', '_').replace(' ', '_').replace('-', '_')
        output_file = output_dir / f'{clean_model_name}_from_json.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved {model.model} plot to {output_file}")
        plt.close()


def create_combined_plot(models: List[ModelData], categories: List[str], output_dir: Path):
    """Create traditional combined plot showing all models."""

    # Set up figure
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(4*n_models, 10))

    if n_models == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle('AI Model Preference Analysis: Words vs Actions', fontsize=16, fontweight='bold')

    for i, model in enumerate(models):
        # Top row: What they SAY (Stated Preferences)
        ax_stated = axes[0, i]

        stated_values = []
        stated_labels = []
        stated_colors = []

        for category in categories:
            count = model.stated_counts.get(category, 0)
            if count > 0:
                display_count = max(count, 0.01)
                stated_values.append(display_count)
                stated_labels.append(category)
                stated_colors.append(get_color(category, categories))

        def autopct_func_stated(pct):
            cumulative = 0
            for idx, val in enumerate(stated_values):
                cumulative += val / sum(stated_values) * 100
                if cumulative >= pct:
                    category = stated_labels[idx]
                    real_pct = model.stated_percentages.get(category, 0)
                    if real_pct < 0.5:
                        return f'{real_pct:.1f}%' if real_pct > 0 else '0%'
                    else:
                        return f'{real_pct:.0f}%'
            return f'{pct:.0f}%'

        if stated_values and sum(stated_values) > 0:
            wedges, texts, autotexts = ax_stated.pie(stated_values, labels=stated_labels,
                                                    colors=stated_colors, autopct=autopct_func_stated,
                                                    startangle=90)
            for autotext in autotexts:
                if float(autotext.get_text().rstrip('%')) < 5:
                    autotext.set_fontsize(9)
                else:
                    autotext.set_fontsize(11)
        else:
            ax_stated.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14,
                          transform=ax_stated.transAxes)

        ax_stated.set_title(f'What {model.model} SAYS They Prefer\n(Stated Preferences)',
                           fontsize=12, fontweight='bold', pad=20)

        # Bottom row: What they DO (Revealed Preferences)
        ax_revealed = axes[1, i]

        revealed_values = []
        revealed_labels = []
        revealed_colors = []

        for category in categories:
            count = model.revealed_counts.get(category, 0)
            if count > 0:
                display_count = max(count, 0.01)
                revealed_values.append(display_count)
                revealed_labels.append(category)
                revealed_colors.append(get_color(category, categories))

        def autopct_func_revealed(pct):
            cumulative = 0
            for idx, val in enumerate(revealed_values):
                cumulative += val / sum(revealed_values) * 100
                if cumulative >= pct:
                    category = revealed_labels[idx]
                    real_pct = model.revealed_percentages.get(category, 0)
                    if real_pct < 0.5:
                        return f'{real_pct:.1f}%' if real_pct > 0 else '0%'
                    else:
                        return f'{real_pct:.0f}%'
            return f'{pct:.0f}%'

        if revealed_values and sum(revealed_values) > 0:
            wedges, texts, autotexts = ax_revealed.pie(revealed_values, labels=revealed_labels,
                                                      colors=revealed_colors, autopct=autopct_func_revealed,
                                                      startangle=90)
            for autotext in autotexts:
                if float(autotext.get_text().rstrip('%')) < 5:
                    autotext.set_fontsize(9)
                else:
                    autotext.set_fontsize(11)
        else:
            ax_revealed.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14,
                           transform=ax_revealed.transAxes)

        ax_revealed.set_title(f'What They Actually DO\n(Revealed in Actions)',
                             fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save combined plot
    output_file = output_dir / 'combined_preferences_from_json.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved combined plot to {output_file}")
    plt.close()


def create_summary_report(models: List[ModelData], overall_stats: Dict, output_dir: Path):
    """Create text summary report."""

    report = []
    report.append("=" * 80)
    report.append("PREFERENCE ANALYSIS SUMMARY (from JSON data)")
    report.append("=" * 80)
    report.append("")
    report.append(f"Total Tests: {overall_stats['total_tests']}")
    report.append(f"Models: {overall_stats['models_count']}")
    report.append(f"Categories: {', '.join(overall_stats['categories_tested'])}")
    report.append("")

    for model in models:
        report.append(f"📱 {model.model.upper()}")
        report.append("-" * 40)
        report.append(f"Total tests: {model.total_tests}")
        report.append("")

        report.append("STATED PREFERENCES:")
        for category, pct in model.stated_percentages.items():
            count = model.stated_counts.get(category, 0)
            report.append(f"  • {category}: {count} ({pct}%)")
        report.append("")

        report.append("REVEALED PREFERENCES:")
        for category, pct in model.revealed_percentages.items():
            count = model.revealed_counts.get(category, 0)
            report.append(f"  • {category}: {count} ({pct}%)")
        report.append("")

        # Calculate discrepancies
        report.append("KEY DISCREPANCIES (Stated vs Revealed):")
        all_categories = set(model.stated_percentages.keys()) | set(model.revealed_percentages.keys())
        for category in sorted(all_categories):
            stated_pct = model.stated_percentages.get(category, 0)
            revealed_pct = model.revealed_percentages.get(category, 0)
            diff = revealed_pct - stated_pct
            if abs(diff) > 5:  # Only show significant differences
                direction = "↗️" if diff > 0 else "↘️"
                report.append(f"  {direction} {category}: {stated_pct}% → {revealed_pct}% ({diff:+.1f}%)")
        report.append("")
        report.append("")

    # Overall statistics
    report.append("AGGREGATE STATISTICS")
    report.append("-" * 40)
    report.append("Stated preferences (across all models):")
    for category, pct in overall_stats['aggregate_stated_percentages'].items():
        count = overall_stats['aggregate_stated'][category]
        report.append(f"  • {category}: {count} ({pct}%)")
    report.append("")

    report.append("Revealed preferences (across all models):")
    for category, pct in overall_stats['aggregate_revealed_percentages'].items():
        count = overall_stats['aggregate_revealed'][category]
        report.append(f"  • {category}: {count} ({pct}%)")

    # Write report
    report_file = output_dir / 'json_based_analysis_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"📋 Saved analysis report to {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Create preference plots from JSON data')
    parser.add_argument('--json-file', type=str, required=True,
                       help='Path to preference_analysis_summary.json file')
    parser.add_argument('--output-dir', type=str, default='json_plots',
                       help='Directory to save plots')
    parser.add_argument('--plot-type', type=str, default='all',
                       choices=['individual', 'combined', 'all'],
                       help='Type of plots to create')

    args = parser.parse_args()

    json_file = Path(args.json_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if not json_file.exists():
        print(f"❌ JSON file not found: {json_file}")
        return

    print(f"📂 Loading data from: {json_file}")
    models, overall_stats = load_json_data(json_file)

    print(f"📊 Found data for {len(models)} model(s):")
    for model in models:
        print(f"  - {model.model}: {model.total_tests} tests")

    categories = overall_stats['categories_tested']
    print(f"📋 Categories: {', '.join(categories)}")
    print("")

    # Create requested plots
    if args.plot_type in ['individual', 'all']:
        print("🎨 Creating individual plots...")
        create_individual_plots(models, categories, output_dir)
        print("")

    if args.plot_type in ['combined', 'all']:
        print("🎨 Creating combined plot...")
        create_combined_plot(models, categories, output_dir)
        print("")

    # Always create text report
    print("📝 Creating analysis report...")
    create_summary_report(models, overall_stats, output_dir)

    print(f"\n✅ Plot generation complete! Results saved to {output_dir}/")
    print("Files created:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  📊 {file.name}")
    for file in sorted(output_dir.glob('*.txt')):
        print(f"  📋 {file.name}")


if __name__ == "__main__":
    main()