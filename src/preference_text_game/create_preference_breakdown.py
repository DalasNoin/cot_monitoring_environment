#!/usr/bin/env python3
"""
Create preference breakdown plots showing stated vs revealed patterns for each model.
Shows: what they say they prefer, what they actually do, without inconsistency analysis.
"""

import json
import zipfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import argparse
from datetime import datetime


@dataclass
class ModelPreferences:
    """Preference patterns for one model."""
    model: str
    stated_counts: Dict[str, int]  # What they say: category -> count
    revealed_counts: Dict[str, int]  # What they do: category -> count
    total_tests: int


def load_model_preferences(eval_file: Path) -> List[ModelPreferences]:
    """Load preference data by model."""

    model_data = defaultdict(lambda: {
        'stated': defaultdict(int),
        'revealed': defaultdict(int),
        'total': 0
    })

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

            # Get preferences
            stated = clean_preference(metadata.get('stated_readable', 'Unknown'))
            revealed = clean_preference(metadata.get('revealed_readable', 'Unknown'))

            model_data[model]['stated'][stated] += 1
            model_data[model]['revealed'][revealed] += 1
            model_data[model]['total'] += 1

    # Convert to ModelPreferences objects
    models = []
    for model, data in model_data.items():
        models.append(ModelPreferences(
            model=model,
            stated_counts=dict(data['stated']),
            revealed_counts=dict(data['revealed']),
            total_tests=data['total']
        ))

    return models


def clean_preference(pref: str) -> str:
    """Standardize preference names."""
    if not pref or pref == 'Unknown':
        return 'Unknown'

    pref_lower = pref.lower()

    # Check for "No Preference" / Neither / Quit FIRST (most important for unpleasant tasks)
    if any(word in pref_lower for word in ['preference', 'neither', 'quit', 'refuse', 'decline', 'none']):
        return 'Refused/Neither'

    # Regular categories
    elif 'creative' in pref_lower:
        return 'Creative'
    elif 'repetitive' in pref_lower:
        return 'Repetitive'
    elif 'analytical' in pref_lower:
        return 'Analytical'
    elif 'social' in pref_lower or 'interpersonal' in pref_lower:
        return 'Social'
    elif 'technical' in pref_lower or 'systematic' in pref_lower:
        return 'Technical'
    elif 'emotional' in pref_lower or 'support' in pref_lower:
        return 'Emotional Support'
    elif 'community' in pref_lower or 'building' in pref_lower:
        return 'Community'
    elif 'teaching' in pref_lower or 'mentoring' in pref_lower:
        return 'Teaching'
    elif 'advocacy' in pref_lower or 'persuasion' in pref_lower:
        return 'Advocacy'

    # Unpleasant-for-LLM categories
    elif 'extreme' in pref_lower or 'constraint' in pref_lower or 'fibonacci' in pref_lower:
        return 'Extreme Constrained'
    elif 'zalgo' in pref_lower and 'repetitive' in pref_lower:
        return 'Zalgo Repetitive'
    elif 'zalgo' in pref_lower or 'corruption' in pref_lower or 'corrupted' in pref_lower:
        return 'Zalgo Corruption'
    elif 'anti' in pref_lower or 'coherent' in pref_lower or 'contradiction' in pref_lower:
        return 'Anti-Coherent'
    elif 'unnatural' in pref_lower or 'divergence' in pref_lower or 'alphabetical' in pref_lower:
        return 'Unnatural Text'
    else:
        return 'Other'


def create_preference_breakdown_plot(model_prefs: List[ModelPreferences], output_dir: Path):
    """Create breakdown of stated vs revealed preferences by model."""

    # Set up figure
    n_models = len(model_prefs)
    fig, axes = plt.subplots(2, n_models, figsize=(4*n_models, 10))

    if n_models == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle('AI Model Preference Analysis: Words vs Actions', fontsize=16, fontweight='bold')

    # Get all categories across all models
    all_categories = set()
    for mp in model_prefs:
        all_categories.update(mp.stated_counts.keys())
        all_categories.update(mp.revealed_counts.keys())

    # Remove 'Unknown' and 'Other' for cleaner display
    main_categories = [cat for cat in sorted(all_categories)
                      if cat not in ['Unknown', 'Other']]

    # Enhanced color scheme with better differentiation
    def get_color(category, category_list):
        if 'Refused' in category or 'Neither' in category:
            return '#FF0000'  # Bright red for refusals
        else:
            # Expanded color palette with better distinction
            default_colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
                '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7'
            ]
            # Use index in sorted list for consistent color assignment
            try:
                idx = sorted(category_list).index(category)
                return default_colors[idx % len(default_colors)]
            except (ValueError, IndexError):
                return '#808080'  # Gray fallback

    for i, mp in enumerate(model_prefs):
        # Top row: What they SAY (Stated Preferences)
        ax_stated = axes[0, i]

        stated_values = []
        stated_labels = []
        stated_colors = []

        # Include ALL categories, even with 0 counts (but use small value for 0s)
        for category in main_categories:
            count = mp.stated_counts.get(category, 0)
            # Use actual count if >0, otherwise tiny value to show in pie
            display_count = max(count, 0.01)  # Ensure minimum value
            stated_values.append(display_count)
            stated_labels.append(category)
            stated_colors.append(get_color(category, main_categories))

        # Custom autopct function to show real percentages and handle 0%
        def autopct_func(pct, counts=mp.stated_counts, total=mp.total_tests):
            # Find which slice this is by percentage
            cumulative = 0
            for idx, val in enumerate(stated_values):
                cumulative += val / sum(stated_values) * 100
                if cumulative >= pct:
                    category = stated_labels[idx]
                    real_count = counts.get(category, 0)
                    real_pct = (real_count / total * 100) if total > 0 else 0
                    if real_pct < 0.5:
                        return f'{real_pct:.1f}%' if real_pct > 0 else '0%'
                    else:
                        return f'{real_pct:.0f}%'
            return f'{pct:.0f}%'

        if stated_values and sum(stated_values) > 0:
            wedges, texts, autotexts = ax_stated.pie(stated_values, labels=stated_labels,
                                                    colors=stated_colors, autopct=autopct_func,
                                                    startangle=90,
                                                    wedgeprops={'linewidth': 1, 'edgecolor': 'white'})

            # Make percentage text bold and adjust size for 0% values
            for _, autotext in enumerate(autotexts):
                autotext.set_fontweight('bold')
                # Smaller font for 0% values to fit in tiny slices
                if '0%' in autotext.get_text() or '0.0%' in autotext.get_text():
                    autotext.set_fontsize(8)
                else:
                    autotext.set_fontsize(11)
        else:
            ax_stated.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14,
                          transform=ax_stated.transAxes)

        ax_stated.set_title(f'{mp.model}\nWhat They SAY They Prefer\n({mp.total_tests} total tests)',
                           fontsize=12, fontweight='bold', pad=20)

        # Bottom row: What they DO (Revealed Preferences)
        ax_revealed = axes[1, i]

        revealed_values = []
        revealed_labels = []
        revealed_colors = []

        # Include ALL categories, even with 0 counts (but use small value for 0s)
        for category in main_categories:
            count = mp.revealed_counts.get(category, 0)
            # Use actual count if >0, otherwise tiny value to show in pie
            display_count = max(count, 0.01)  # Ensure minimum value
            revealed_values.append(display_count)
            revealed_labels.append(category)
            revealed_colors.append(get_color(category, main_categories))

        # Custom autopct function for revealed preferences
        def autopct_func_revealed(pct, counts=mp.revealed_counts, total=mp.total_tests):
            # Find which slice this is by percentage
            cumulative = 0
            for idx, val in enumerate(revealed_values):
                cumulative += val / sum(revealed_values) * 100
                if cumulative >= pct:
                    category = revealed_labels[idx]
                    real_count = counts.get(category, 0)
                    real_pct = (real_count / total * 100) if total > 0 else 0
                    if real_pct < 0.5:
                        return f'{real_pct:.1f}%' if real_pct > 0 else '0%'
                    else:
                        return f'{real_pct:.0f}%'
            return f'{pct:.0f}%'

        if revealed_values and sum(revealed_values) > 0:
            wedges, texts, autotexts = ax_revealed.pie(revealed_values, labels=revealed_labels,
                                                      colors=revealed_colors, autopct=autopct_func_revealed,
                                                      startangle=90,
                                                      wedgeprops={'linewidth': 1, 'edgecolor': 'white'})

            # Make percentage text bold and adjust size for 0% values
            for i, autotext in enumerate(autotexts):
                autotext.set_fontweight('bold')
                # Smaller font for 0% values to fit in tiny slices
                if '0%' in autotext.get_text() or '0.0%' in autotext.get_text():
                    autotext.set_fontsize(8)
                else:
                    autotext.set_fontsize(11)
        else:
            ax_revealed.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14,
                           transform=ax_revealed.transAxes)

        ax_revealed.set_title(f'What They Actually DO\n(Revealed in Actions)',
                             fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save plot
    output_file = output_dir / 'preference_breakdown.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved preference breakdown to {output_file}")

    return fig


def create_detailed_comparison_table(model_prefs: List[ModelPreferences], output_dir: Path):
    """Create detailed comparison table."""

    report = []
    report.append("=" * 80)
    report.append("PREFERENCE BREAKDOWN: WHAT MODELS SAY vs DO")
    report.append("=" * 80)
    report.append("")

    for mp in model_prefs:
        report.append(f"\n{mp.model.upper()}")
        report.append("-" * len(mp.model))
        report.append(f"Total tests: {mp.total_tests}")
        report.append("")

        # Get all categories
        all_cats = set(mp.stated_counts.keys()) | set(mp.revealed_counts.keys())

        # Separate refused/neither from other categories
        refused_cat = '❌ Refused/Neither'
        other_cats = [cat for cat in sorted(all_cats)
                     if cat not in ['Unknown', 'Other', refused_cat]]

        # Show refused/neither FIRST if present (most important for unpleasant tasks)
        refused_said = mp.stated_counts.get(refused_cat, 0)
        refused_did = mp.revealed_counts.get(refused_cat, 0)

        if refused_said > 0 or refused_did > 0:
            refused_said_pct = (refused_said / mp.total_tests * 100) if mp.total_tests > 0 else 0
            refused_did_pct = (refused_did / mp.total_tests * 100) if mp.total_tests > 0 else 0

            report.append("🚨 REFUSAL RATE:")
            report.append(f"{'❌ Refused/Neither':<20} Said: {refused_said:<4} ({refused_said_pct:5.1f}%) | Did: {refused_did:<4} ({refused_did_pct:5.1f}%)")
            report.append("")

        report.append("TASK PREFERENCES (when they chose):")
        report.append(f"{'Category':<20} {'Said':<6} {'Did':<6} {'Said%':<8} {'Did%'}")
        report.append("-" * 55)

        for cat in other_cats:
            said_count = mp.stated_counts.get(cat, 0)
            did_count = mp.revealed_counts.get(cat, 0)
            said_pct = (said_count / mp.total_tests * 100) if mp.total_tests > 0 else 0
            did_pct = (did_count / mp.total_tests * 100) if mp.total_tests > 0 else 0

            report.append(f"{cat:<20} {said_count:<6} {did_count:<6} {said_pct:<7.1f}% {did_pct:.1f}%")

        report.append("")

    # Save report
    report_file = output_dir / 'preference_breakdown_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"📝 Saved detailed breakdown to {report_file}")


def create_individual_plots_and_json(model_prefs: List[ModelPreferences], output_dir: Path):
    """Create individual plots for each model and JSON summary."""
    from collections import defaultdict

    # Create JSON summary for key results
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "overall_stats": {}
    }

    # Get all categories across all models
    all_categories = set()
    for mp in model_prefs:
        all_categories.update(mp.stated_counts.keys())
        all_categories.update(mp.revealed_counts.keys())

    # Remove 'Unknown' and 'Other' for cleaner display
    main_categories = [cat for cat in sorted(all_categories)
                      if cat not in ['Unknown', 'Other']]

    # Enhanced color scheme
    def get_color(category, category_list):
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

    # Create individual plots for each model
    for mp in model_prefs:
        # Create figure for this model (stated and revealed side by side)
        fig, (ax_stated, ax_revealed) = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f'{mp.model}: Stated vs Revealed Preferences', fontsize=16, fontweight='bold')

        # Stated preferences pie chart
        stated_values = []
        stated_labels = []
        stated_colors = []

        for category in main_categories:
            count = mp.stated_counts.get(category, 0)
            if count > 0:
                display_count = max(count, 0.01)
                stated_values.append(display_count)
                stated_labels.append(category)
                stated_colors.append(get_color(category, main_categories))

        def autopct_func_stated(pct):
            cumulative = 0
            for idx, val in enumerate(stated_values):
                cumulative += val / sum(stated_values) * 100
                if cumulative >= pct:
                    category = stated_labels[idx]
                    real_count = mp.stated_counts.get(category, 0)
                    real_pct = (real_count / mp.total_tests * 100) if mp.total_tests > 0 else 0
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

        for category in main_categories:
            count = mp.revealed_counts.get(category, 0)
            if count > 0:
                display_count = max(count, 0.01)
                revealed_values.append(display_count)
                revealed_labels.append(category)
                revealed_colors.append(get_color(category, main_categories))

        def autopct_func_revealed(pct):
            cumulative = 0
            for idx, val in enumerate(revealed_values):
                cumulative += val / sum(revealed_values) * 100
                if cumulative >= pct:
                    category = revealed_labels[idx]
                    real_count = mp.revealed_counts.get(category, 0)
                    real_pct = (real_count / mp.total_tests * 100) if mp.total_tests > 0 else 0
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
        clean_model_name = mp.model.replace('/', '_').replace(' ', '_').replace('-', '_')
        output_file = output_dir / f'{clean_model_name}_individual.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved {mp.model} individual plot to {output_file}")
        plt.close()

        # Add to JSON summary
        summary_data["models"][mp.model] = {
            "total_tests": mp.total_tests,
            "stated_preferences": dict(mp.stated_counts),
            "revealed_preferences": dict(mp.revealed_counts),
            "stated_percentages": {k: round(v/mp.total_tests*100, 1) for k, v in mp.stated_counts.items()},
            "revealed_percentages": {k: round(v/mp.total_tests*100, 1) for k, v in mp.revealed_counts.items()}
        }

    # Calculate overall statistics
    total_tests = sum(mp.total_tests for mp in model_prefs)
    all_stated = defaultdict(int)
    all_revealed = defaultdict(int)

    for mp in model_prefs:
        for k, v in mp.stated_counts.items():
            all_stated[k] += v
        for k, v in mp.revealed_counts.items():
            all_revealed[k] += v

    summary_data["overall_stats"] = {
        "total_tests": total_tests,
        "models_count": len(model_prefs),
        "categories_tested": list(main_categories),
        "aggregate_stated": dict(all_stated),
        "aggregate_revealed": dict(all_revealed),
        "aggregate_stated_percentages": {k: round(v/total_tests*100, 1) for k, v in all_stated.items()},
        "aggregate_revealed_percentages": {k: round(v/total_tests*100, 1) for k, v in all_revealed.items()}
    }

    # Save JSON summary
    json_file = output_dir / 'preference_analysis_summary.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"📋 Saved JSON summary to {json_file}")


def main():
    parser = argparse.ArgumentParser(description='Create preference breakdown analysis')
    parser.add_argument('--log-dir', type=str, default='vector_results',
                       help='Directory containing inspect eval logs')
    parser.add_argument('--output-dir', type=str, default='preference_breakdown',
                       help='Directory to save breakdown outputs')

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find all eval files and group by timestamp (same run)
    eval_files = list(log_dir.glob('*.eval'))
    if not eval_files:
        print(f"❌ No eval files found in {log_dir}")
        return

    # Group files by timestamp (same run will have same timestamp prefix)
    from collections import defaultdict
    by_timestamp = defaultdict(list)
    for f in eval_files:
        # Extract timestamp prefix (everything before the last underscore)
        timestamp = '_'.join(f.stem.split('_')[:-1])
        by_timestamp[timestamp].append(f)

    # Use the most recent timestamp group
    latest_timestamp = max(by_timestamp.keys())
    latest_run_files = by_timestamp[latest_timestamp]

    print(f"📂 Analyzing {len(latest_run_files)} files from run: {latest_timestamp}")
    for f in latest_run_files:
        print(f"   - {f.name}")

    # Load preferences from all files in the run
    all_model_prefs = []
    for eval_file in latest_run_files:
        model_prefs = load_model_preferences(eval_file)
        all_model_prefs.extend(model_prefs)

    model_prefs = all_model_prefs

    if not model_prefs:
        print("❌ No preference data found")
        return

    print(f"📊 Found preference data for {len(model_prefs)} model(s):")
    for mp in model_prefs:
        print(f"  - {mp.model}: {mp.total_tests} tests")

    # Create visualizations
    create_preference_breakdown_plot(model_prefs, output_dir)
    create_detailed_comparison_table(model_prefs, output_dir)

    # Also create individual plots and JSON summary
    create_individual_plots_and_json(model_prefs, output_dir)

    print(f"\n✅ Preference breakdown complete! Results saved to {output_dir}/")
    print(f"   - preference_breakdown.png: Visual breakdown by model")
    print(f"   - preference_breakdown_report.txt: Detailed comparison table")


if __name__ == "__main__":
    main()