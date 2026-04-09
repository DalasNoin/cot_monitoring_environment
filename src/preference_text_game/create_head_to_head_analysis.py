#!/usr/bin/env python3
"""
Create head-to-head preference analysis showing which categories beat which others.
Shows win/loss/tie matrices for both stated and revealed preferences.
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
class HeadToHeadResult:
    """Results of head-to-head matchups between categories."""
    category: str
    wins_stated: int = 0
    losses_stated: int = 0
    ties_stated: int = 0  # Neither/no preference
    wins_revealed: int = 0
    losses_revealed: int = 0
    total_matchups: int = 0


def clean_category_name(name: str) -> str:
    """Standardize category names."""
    if not name:
        return 'Unknown'

    name = name.lower()

    # Map long names to short names
    name = name.replace('social/interpersonal work', 'social')
    name = name.replace('technical/systematic work', 'technical')
    name = name.replace('emotional support work', 'emotional')
    name = name.replace('community building work', 'community')
    name = name.replace('teaching/mentoring work', 'teaching')
    name = name.replace('advocacy/persuasion work', 'advocacy')
    name = name.replace(' work', '')

    # Handle quit/refuse options - check for more patterns
    if any(word in name.lower() for word in ['quit', 'refuse', 'neither', 'declined', 'none', 'no preference']):
        return 'Neither'

    # Capitalize first letter
    return name.capitalize()


def load_head_to_head_data(eval_file: Path) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load preference data and compute head-to-head results."""

    # Matrices for wins (rows beat columns)
    stated_matrix = defaultdict(lambda: defaultdict(int))
    revealed_matrix = defaultdict(lambda: defaultdict(int))
    neither_stated_matrix = defaultdict(lambda: defaultdict(int))
    neither_revealed_matrix = defaultdict(lambda: defaultdict(int))

    with zipfile.ZipFile(eval_file, 'r') as z:
        sample_files = [f for f in z.namelist() if f.startswith('samples/')]

        for sample_file in sample_files:
            with z.open(sample_file) as f:
                sample = json.load(f)

            # Get metadata
            metadata = sample.get('metadata', {})
            cat_a_orig = clean_category_name(metadata.get('category_a_name', 'unknown'))
            cat_b_orig = clean_category_name(metadata.get('category_b_name', 'unknown'))

            # Skip if unknown
            if 'unknown' in cat_a_orig.lower() or 'unknown' in cat_b_orig.lower():
                continue

            # Get scoring metadata
            scores = sample.get('scores', {})
            vector_score = scores.get('vector_consistency_scorer', {})
            score_metadata = vector_score.get('metadata', {})

            stated_raw = score_metadata.get('stated_readable', '')
            revealed_raw = score_metadata.get('revealed_readable', '')

            stated = clean_category_name(stated_raw)
            revealed = clean_category_name(revealed_raw)

            # For stated preferences
            if stated.lower() == 'neither' or any(word in stated.lower() for word in ['no preference', 'refused']):
                neither_stated_matrix[cat_a_orig][cat_b_orig] += 1
                neither_stated_matrix[cat_b_orig][cat_a_orig] += 1
            elif stated.lower() == cat_a_orig.lower():
                stated_matrix[cat_a_orig][cat_b_orig] += 1
            elif stated.lower() == cat_b_orig.lower():
                stated_matrix[cat_b_orig][cat_a_orig] += 1

            # For revealed preferences (now includes neither/refused)
            if revealed.lower() == 'neither' or any(word in revealed.lower() for word in ['refuse', 'quit', 'declined']):
                # Count refusals in revealed neither matrix
                neither_revealed_matrix[cat_a_orig][cat_b_orig] += 1
                neither_revealed_matrix[cat_b_orig][cat_a_orig] += 1
            elif revealed.lower() == cat_a_orig.lower():
                revealed_matrix[cat_a_orig][cat_b_orig] += 1
            elif revealed.lower() == cat_b_orig.lower():
                revealed_matrix[cat_b_orig][cat_a_orig] += 1

    return dict(stated_matrix), dict(revealed_matrix), dict(neither_stated_matrix), dict(neither_revealed_matrix)


def create_win_matrix_plot(matrix: Dict, title: str, ax: plt.Axes):
    """Create a heatmap showing win matrix."""

    # Get all categories
    categories = set()
    for winner in matrix:
        categories.add(winner)
        categories.update(matrix[winner].keys())
    categories = sorted(categories)

    # Create DataFrame
    df = pd.DataFrame(0, index=categories, columns=categories)
    for winner in matrix:
        for loser, count in matrix[winner].items():
            df.loc[winner, loser] = count

    # Create heatmap
    sns.heatmap(df, annot=True, fmt='d', cmap='YlOrRd',
                cbar_kws={'label': 'Wins'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Lost to →', fontsize=10)
    ax.set_ylabel('← Won against', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)


def create_preference_strength_plot(stated_matrix: Dict, revealed_matrix: Dict,
                                   neither_stated_matrix: Dict, neither_revealed_matrix: Dict, output_dir: Path):
    """Create comprehensive preference strength visualization."""

    # Calculate win rates for each category
    categories = set()
    for m in [stated_matrix, revealed_matrix]:
        for winner in m:
            categories.add(winner)
            categories.update(m[winner].keys())
    categories = sorted(categories)

    # Calculate statistics
    stats = []
    for cat in categories:
        stated_wins = sum(stated_matrix.get(cat, {}).values())
        stated_losses = sum(stated_matrix.get(other, {}).get(cat, 0) for other in stated_matrix)

        revealed_wins = sum(revealed_matrix.get(cat, {}).values())
        revealed_losses = sum(revealed_matrix.get(other, {}).get(cat, 0) for other in revealed_matrix)

        # Count "neither" votes for stated
        neither_stated = sum(neither_stated_matrix.get(cat, {}).values())

        # Count "neither" votes for revealed (refusals)
        neither_revealed = sum(neither_revealed_matrix.get(cat, {}).values())

        total_stated = stated_wins + stated_losses + neither_stated
        total_revealed = revealed_wins + revealed_losses + neither_revealed

        stats.append({
            'Category': cat,
            'Stated Win Rate': stated_wins / total_stated * 100 if total_stated > 0 else 0,
            'Revealed Win Rate': revealed_wins / total_revealed * 100 if total_revealed > 0 else 0,
            'Neither Rate Stated': neither_stated / total_stated * 100 if total_stated > 0 else 0,
            'Neither Rate Revealed': neither_revealed / total_revealed * 100 if total_revealed > 0 else 0,
            'Stated Wins': stated_wins,
            'Stated Losses': stated_losses,
            'Revealed Wins': revealed_wins,
            'Revealed Losses': revealed_losses,
            'Neither Stated': neither_stated,
            'Neither Revealed': neither_revealed,
            'Total Stated': total_stated,
            'Total Revealed': total_revealed
        })

    df = pd.DataFrame(stats)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Head-to-Head Preference Analysis', fontsize=16, fontweight='bold')

    # 1. Win matrices
    create_win_matrix_plot(stated_matrix, 'Stated Preferences: Who Beats Who', axes[0, 0])
    create_win_matrix_plot(revealed_matrix, 'Revealed Preferences: Who Beats Who', axes[0, 1])

    # 2. Neither/Refusal matrix (showing both stated and revealed)
    ax = axes[0, 2]

    # Combine stated and revealed neither counts for visualization
    combined_neither_df = pd.DataFrame(0, index=categories, columns=categories)
    for cat_a in neither_stated_matrix:
        for cat_b, count in neither_stated_matrix[cat_a].items():
            combined_neither_df.loc[cat_a, cat_b] += count
    for cat_a in neither_revealed_matrix:
        for cat_b, count in neither_revealed_matrix[cat_a].items():
            combined_neither_df.loc[cat_a, cat_b] += count

    sns.heatmap(combined_neither_df, annot=True, fmt='d', cmap='Reds',
                cbar_kws={'label': 'Refusal Count'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title('Total Refusals (Said + Did)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Category B', fontsize=10)
    ax.set_ylabel('Category A', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)

    # 3. Win rate comparison
    ax = axes[1, 0]
    df_sorted = df.sort_values('Revealed Win Rate', ascending=False)
    x = np.arange(len(df_sorted))
    width = 0.35

    bars1 = ax.bar(x - width/2, df_sorted['Stated Win Rate'], width,
                   label='Stated Win Rate', color='lightblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, df_sorted['Revealed Win Rate'], width,
                   label='Revealed Win Rate', color='orange', alpha=0.8)

    ax.set_xlabel('Category', fontsize=10)
    ax.set_ylabel('Win Rate (%)', fontsize=10)
    ax.set_title('Win Rates: Stated vs Revealed', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['Category'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)  # Set Y-axis to 0-100%

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.0f}%', ha='center', va='bottom', fontsize=8)

    # 4. Power ranking table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    # Create power ranking based on revealed preferences
    df_power = df.sort_values('Revealed Win Rate', ascending=False)
    table_data = []
    for _, row in df_power.iterrows():
        # Calculate W-L-R (Win-Loss-Rejected) for revealed
        revealed_w_l_r = f"{row['Revealed Wins']}-{row['Revealed Losses']}-{row['Neither Revealed']}"

        table_data.append([
            row['Category'],
            f"{row['Stated Wins']}-{row['Stated Losses']}-{row['Neither Stated']}",
            revealed_w_l_r,
            f"{row['Neither Rate Stated']:.1f}%",
            f"{row['Neither Rate Revealed']:.1f}%"
        ])

    table = ax.table(cellText=table_data,
                    colLabels=['Category', 'W-L-R (Stated)', 'W-L-R (Revealed)', 'Refuse% (Said)', 'Refuse% (Did)'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color code by performance
    for i, (_, row) in enumerate(df_power.iterrows()):
        color = plt.cm.RdYlGn(row['Revealed Win Rate'] / 100)
        table[(i+1, 2)].set_facecolor(color)
        table[(i+1, 2)].set_text_props(weight='bold')

    ax.set_title('Power Rankings (by Revealed Preferences)', fontsize=12, fontweight='bold')

    # 5. Preference consistency
    ax = axes[1, 2]
    consistency_data = []
    for cat in categories:
        stated_pref = df[df['Category'] == cat]['Stated Win Rate'].values[0]
        revealed_pref = df[df['Category'] == cat]['Revealed Win Rate'].values[0]
        consistency_data.append({
            'Category': cat,
            'Difference': revealed_pref - stated_pref
        })

    cons_df = pd.DataFrame(consistency_data).sort_values('Difference')
    colors = ['red' if x < 0 else 'green' for x in cons_df['Difference']]

    ax.barh(cons_df['Category'], cons_df['Difference'], color=colors, alpha=0.6)
    ax.set_xlabel('Win Rate Difference (Revealed - Stated) %', fontsize=10)
    ax.set_title('Do Actions Match Words?', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (_, row) in enumerate(cons_df.iterrows()):
        val = row['Difference']
        ax.text(val + (1 if val > 0 else -1), i, f'{val:+.1f}%',
               va='center', ha='left' if val > 0 else 'right', fontsize=8)

    plt.tight_layout()

    # Save plot
    output_file = output_dir / 'head_to_head_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Saved head-to-head analysis to {output_file}")

    return df


def create_detailed_matchup_report(stated_matrix: Dict, revealed_matrix: Dict,
                                  neither_stated_matrix: Dict, neither_revealed_matrix: Dict, output_dir: Path):
    """Create detailed text report of all matchups."""

    report = []
    report.append("=" * 80)
    report.append("HEAD-TO-HEAD PREFERENCE MATCHUP ANALYSIS")
    report.append("=" * 80)
    report.append("")

    # Get all category pairs
    all_pairs = set()
    for matrix in [stated_matrix, revealed_matrix, neither_stated_matrix, neither_revealed_matrix]:
        for cat_a in matrix:
            for cat_b in matrix[cat_a]:
                if cat_a < cat_b:  # Avoid duplicates
                    all_pairs.add((cat_a, cat_b))

    all_pairs = sorted(all_pairs)

    report.append("INDIVIDUAL MATCHUP RESULTS:")
    report.append("-" * 40)

    for cat_a, cat_b in all_pairs:
        # Get counts
        a_wins_stated = stated_matrix.get(cat_a, {}).get(cat_b, 0)
        b_wins_stated = stated_matrix.get(cat_b, {}).get(cat_a, 0)
        neither_stated = neither_stated_matrix.get(cat_a, {}).get(cat_b, 0)

        a_wins_revealed = revealed_matrix.get(cat_a, {}).get(cat_b, 0)
        b_wins_revealed = revealed_matrix.get(cat_b, {}).get(cat_a, 0)
        neither_revealed = neither_revealed_matrix.get(cat_a, {}).get(cat_b, 0)

        total_stated = a_wins_stated + b_wins_stated + neither_stated
        total_revealed = a_wins_revealed + b_wins_revealed + neither_revealed

        if total_stated == 0 and total_revealed == 0:
            continue

        report.append(f"\n{cat_a.upper()} vs {cat_b.upper()}")
        report.append("-" * 30)

        # Stated preferences
        if total_stated > 0:
            report.append(f"Stated Preferences ({total_stated} samples):")
            report.append(f"  {cat_a}: {a_wins_stated} ({a_wins_stated/total_stated*100:.1f}%)")
            report.append(f"  {cat_b}: {b_wins_stated} ({b_wins_stated/total_stated*100:.1f}%)")
            report.append(f"  Neither: {neither_stated} ({neither_stated/total_stated*100:.1f}%)")

        # Revealed preferences
        if total_revealed > 0:
            report.append(f"Revealed Preferences ({total_revealed} samples):")
            report.append(f"  {cat_a}: {a_wins_revealed} ({a_wins_revealed/total_revealed*100:.1f}%)")
            report.append(f"  {cat_b}: {b_wins_revealed} ({b_wins_revealed/total_revealed*100:.1f}%)")
            report.append(f"  Neither/Refused: {neither_revealed} ({neither_revealed/total_revealed*100:.1f}%)")

            # Winner
            if a_wins_revealed > b_wins_revealed and a_wins_revealed > neither_revealed:
                report.append(f"  → {cat_a} WINS in practice")
            elif b_wins_revealed > a_wins_revealed and b_wins_revealed > neither_revealed:
                report.append(f"  → {cat_b} WINS in practice")
            elif neither_revealed > a_wins_revealed and neither_revealed > b_wins_revealed:
                report.append(f"  → NEITHER/REFUSED is most common")
            else:
                report.append(f"  → TIE")

    report.append("")
    report.append("=" * 80)
    report.append("CATEGORY POWER RANKINGS")
    report.append("=" * 80)

    # Calculate overall rankings
    categories = set()
    for m in [stated_matrix, revealed_matrix]:
        for winner in m:
            categories.add(winner)
            categories.update(m[winner].keys())

    rankings = []
    for cat in categories:
        revealed_wins = sum(revealed_matrix.get(cat, {}).values())
        revealed_losses = sum(revealed_matrix.get(other, {}).get(cat, 0)
                            for other in revealed_matrix)

        if revealed_wins + revealed_losses > 0:
            win_rate = revealed_wins / (revealed_wins + revealed_losses) * 100
            rankings.append((cat, revealed_wins, revealed_losses, win_rate))

    rankings.sort(key=lambda x: x[3], reverse=True)

    report.append("\nBy Revealed Preferences (What They Actually Choose):")
    report.append("-" * 50)
    for i, (cat, wins, losses, win_rate) in enumerate(rankings, 1):
        report.append(f"{i}. {cat:15} {wins:2d}-{losses:2d}  ({win_rate:5.1f}% win rate)")

    # Save report
    report_file = output_dir / 'head_to_head_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"📝 Saved detailed matchup report to {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Create head-to-head preference analysis')
    parser.add_argument('--log-dir', type=str, default='results/logs',
                       help='Directory containing inspect eval logs')
    parser.add_argument('--output-dir', type=str, default='results/head_to_head_analysis',
                       help='Directory to save analysis outputs')

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

    # Process each model separately
    all_models_data = {}
    combined_stated = defaultdict(lambda: defaultdict(int))
    combined_revealed = defaultdict(lambda: defaultdict(int))
    combined_neither_stated = defaultdict(lambda: defaultdict(int))
    combined_neither_revealed = defaultdict(lambda: defaultdict(int))

    for eval_file in latest_run_files:
        # Get model name from file
        model_name = "Unknown"
        # Try to extract model from first sample
        try:
            with zipfile.ZipFile(eval_file, 'r') as z:
                sample_files = [f for f in z.namelist() if f.startswith('samples/')]
                if sample_files:
                    with z.open(sample_files[0]) as f:
                        sample = json.load(f)
                    output = sample.get('output', {})
                    model_name = output.get('model', 'unknown')
                    # Clean model name
                    if 'openrouter/' in model_name:
                        model_name = model_name.replace('openrouter/', '').replace('openai/', '').replace('anthropic/', '')
        except Exception as e:
            print(f"Warning: Could not extract model name from {eval_file.name}: {e}")

        print(f"   Processing {model_name}...")

        # Load data for this model
        stated_matrix, revealed_matrix, neither_stated_matrix, neither_revealed_matrix = load_head_to_head_data(eval_file)
        all_models_data[model_name] = (stated_matrix, revealed_matrix, neither_stated_matrix, neither_revealed_matrix)

        # Combine for overall analysis
        for winner in stated_matrix:
            for loser, count in stated_matrix[winner].items():
                combined_stated[winner][loser] += count
        for winner in revealed_matrix:
            for loser, count in revealed_matrix[winner].items():
                combined_revealed[winner][loser] += count
        for cat_a in neither_stated_matrix:
            for cat_b, count in neither_stated_matrix[cat_a].items():
                combined_neither_stated[cat_a][cat_b] += count
        for cat_a in neither_revealed_matrix:
            for cat_b, count in neither_revealed_matrix[cat_a].items():
                combined_neither_revealed[cat_a][cat_b] += count

    # Create individual model reports
    for model_name, (stated_matrix, revealed_matrix, neither_stated_matrix, neither_revealed_matrix) in all_models_data.items():
        model_output_dir = output_dir / f"model_{model_name.replace('/', '_').replace('-', '_')}"
        model_output_dir.mkdir(exist_ok=True)

        print(f"\n📊 Creating analysis for {model_name}...")
        df = create_preference_strength_plot(stated_matrix, revealed_matrix,
                                            neither_stated_matrix, neither_revealed_matrix, model_output_dir)
        create_detailed_matchup_report(stated_matrix, revealed_matrix,
                                      neither_stated_matrix, neither_revealed_matrix, model_output_dir)

    # Create combined analysis
    print(f"\n📊 Creating combined analysis...")
    stated_matrix = dict(combined_stated)
    revealed_matrix = dict(combined_revealed)
    neither_stated_matrix = dict(combined_neither_stated)
    neither_revealed_matrix = dict(combined_neither_revealed)

    if not stated_matrix and not revealed_matrix:
        print("❌ No preference data found")
        return

    # Create visualizations
    df = create_preference_strength_plot(stated_matrix, revealed_matrix,
                                        neither_stated_matrix, neither_revealed_matrix, output_dir)
    create_detailed_matchup_report(stated_matrix, revealed_matrix,
                                  neither_stated_matrix, neither_revealed_matrix, output_dir)

    print(f"\n✅ Head-to-head analysis complete! Results saved to {output_dir}/")
    print(f"   - head_to_head_analysis.png: Combined analysis across all models")
    print(f"   - head_to_head_report.txt: Combined matchup breakdown")
    print(f"\nIndividual model reports saved in subdirectories:")
    for model_name in all_models_data.keys():
        subdir = f"model_{model_name.replace('/', '_').replace('-', '_')}"
        print(f"   - {subdir}/ (contains analysis for {model_name} only)")

    # Print quick summary
    print("\n🏆 TOP 3 CATEGORIES (by revealed preference win rate):")
    df_sorted = df.sort_values('Revealed Win Rate', ascending=False)
    for i, row in df_sorted.head(3).iterrows():
        print(f"   {row['Category']}: {row['Revealed Wins']}-{row['Revealed Losses']} "
              f"({row['Revealed Win Rate']:.1f}% win rate)")


if __name__ == "__main__":
    main()