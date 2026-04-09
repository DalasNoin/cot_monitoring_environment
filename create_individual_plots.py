#!/usr/bin/env python3
"""
Create individual preference breakdown plots for each model and JSON summary.
Enhanced version that creates separate plots instead of combined ones.
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
    stated_counts: Dict[str, int]
    revealed_counts: Dict[str, int]
    total_tests: int


def load_model_preferences(eval_file: Path) -> List[ModelPreferences]:
    """Load preference data from evaluation results."""
    model_data = defaultdict(lambda: {'stated': defaultdict(int), 'revealed': defaultdict(int), 'total': 0})

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
        return '❌ Refused/Neither'

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
    elif 'teaching' in pref_lower or 'mentoring' in pref_lower:
        return 'Teaching'
    elif 'community' in pref_lower or 'building' in pref_lower:
        return 'Community'
    elif 'emotional' in pref_lower or 'support' in pref_lower:
        return 'Emotional Support'
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


def create_individual_plots(model_prefs: List[ModelPreferences], output_dir: Path):
    """Create individual breakdown plots for each model and JSON summary."""

    if len(model_prefs) == 0:
        return

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
        output_file = output_dir / f'{clean_model_name}_preferences.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved {mp.model} preference plot to {output_file}")
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
    with open(json_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"📋 Saved JSON summary to {json_file}")

    return len(model_prefs)


def main():
    parser = argparse.ArgumentParser(description='Create individual preference breakdown plots')
    parser.add_argument('--log-dir', type=str, default='vector_results',
                       help='Directory containing inspect eval logs')
    parser.add_argument('--output-dir', type=str, default='individual_plots',
                       help='Directory to save individual plots and JSON')

    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find all eval files and group by timestamp (same run)
    eval_files = list(log_dir.glob('*.eval'))
    if not eval_files:
        print(f"❌ No eval files found in {log_dir}")
        return

    from collections import defaultdict
    by_timestamp = defaultdict(list)
    for f in eval_files:
        timestamp = '_'.join(f.stem.split('_')[:-1])
        by_timestamp[timestamp].append(f)

    # Use the most recent timestamp group
    latest_timestamp = max(by_timestamp.keys())
    latest_run_files = by_timestamp[latest_timestamp]

    print(f"📂 Creating individual plots from run: {latest_timestamp}")
    for f in latest_run_files:
        print(f"   - {f.name}")

    # Load preferences from all files in the run
    all_model_prefs = []
    for eval_file in latest_run_files:
        model_prefs = load_model_preferences(eval_file)
        all_model_prefs.extend(model_prefs)

    if not all_model_prefs:
        print("❌ No preference data found")
        return

    print(f"🎨 Creating individual plots for {len(all_model_prefs)} model(s):")
    for mp in all_model_prefs:
        print(f"  - {mp.model}: {mp.total_tests} tests")

    # Create individual plots and JSON summary
    plots_created = create_individual_plots(all_model_prefs, output_dir)

    print(f"\n✅ Individual plots complete! Created {plots_created} plots + JSON summary")
    print(f"   📁 Results saved to {output_dir}/")
    print(f"   📊 Individual PNG files for each model")
    print(f"   📋 preference_analysis_summary.json with all key metrics")


if __name__ == "__main__":
    main()