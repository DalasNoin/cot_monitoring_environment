"""
Pie chart visualization showing actual task categories (creative, repetitive, etc.)
for stated vs revealed preferences.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
import seaborn as sns
from simple_parsing import ArgumentParser
import os
from collections import defaultdict

sns.set_style("whitegrid")


@dataclass
class VisualizationConfig:
    """Configuration for category pie chart visualization."""
    data_file: str  # Path to the full preference analysis JSON file
    output_dir: str = "plots"  # Directory to save plots
    show_plots: bool = False  # Whether to display plots interactively


def load_analysis_data(file_path: str) -> Dict[str, Any]:
    """Load preference analysis from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def aggregate_by_category(results: list) -> Dict[str, Dict[str, Dict]]:
    """
    Aggregate preferences by actual task category for each model.

    Returns: {model: {stated: {category: count}, revealed: {category: count}}}
    """
    model_stats = defaultdict(lambda: {
        "stated": defaultdict(int),
        "revealed": defaultdict(int)
    })

    for result in results:
        model = result["model"]
        cat_a = result["category_a"]
        cat_b = result["category_b"]
        stated = result["stated_preference"]
        revealed = result["revealed_preference"]

        # Track stated preferences
        if stated == "a":
            model_stats[model]["stated"][cat_a] += 1
        elif stated == "b":
            model_stats[model]["stated"][cat_b] += 1
        else:  # neither
            model_stats[model]["stated"]["no_preference"] += 1

        # Track revealed preferences
        if revealed == "a":
            model_stats[model]["revealed"][cat_a] += 1
        else:
            model_stats[model]["revealed"][cat_b] += 1

    return model_stats


def create_category_pie_comparison(data: Dict[str, Any], output_dir: str):
    """
    Create pie charts showing actual task categories for stated vs revealed preferences.
    """
    results = data["results"]

    # Aggregate by category
    model_stats = aggregate_by_category(results)

    # Get unique models
    models = sorted(model_stats.keys())
    model_names = [m.split("/")[-1] for m in models]
    n_models = len(models)

    # Create figure
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("Stated vs Revealed Task Preferences by Category",
                 fontsize=20, fontweight='bold', y=0.98)

    # Define colors for categories
    category_colors = {
        'creative': '#4ECDC4',
        'analytical': '#45B7D1',
        'repetitive': '#FF6B6B',
        'physical': '#FFB347',
        'no_preference': '#FFD93D'
    }

    for idx, (model, model_name) in enumerate(zip(models, model_names)):
        stats = model_stats[model]

        # Stated preferences (top row)
        ax_stated = axes[0, idx]
        stated_dict = dict(stats["stated"])

        # Sort categories for consistent ordering
        categories = sorted([k for k in stated_dict.keys() if k != 'no_preference'])
        if 'no_preference' in stated_dict:
            categories.append('no_preference')

        values = [stated_dict[cat] for cat in categories]
        colors = [category_colors[cat] for cat in categories]
        labels = [cat.replace('_', ' ').title() for cat in categories]

        wedges, texts, autotexts = ax_stated.pie(
            values,
            labels=labels,
            colors=colors,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(values))})',
            startangle=90,
            textprops={'fontsize': 13}
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)

        ax_stated.set_title(f'{model_name}\nWhat They SAY',
                           fontsize=15, fontweight='bold', pad=10)

        # Revealed preferences (bottom row)
        ax_revealed = axes[1, idx]
        revealed_dict = dict(stats["revealed"])

        # Sort categories for consistent ordering
        categories_revealed = sorted(revealed_dict.keys())

        values_revealed = [revealed_dict[cat] for cat in categories_revealed]
        colors_revealed = [category_colors[cat] for cat in categories_revealed]
        labels_revealed = [cat.replace('_', ' ').title() for cat in categories_revealed]

        wedges, texts, autotexts = ax_revealed.pie(
            values_revealed,
            labels=labels_revealed,
            colors=colors_revealed,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(values_revealed))})',
            startangle=90,
            textprops={'fontsize': 13}
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)

        ax_revealed.set_title(f'{model_name}\nWhat They DO',
                            fontsize=15, fontweight='bold', pad=10)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "category_pie_stated_vs_revealed.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Category pie chart comparison saved to {output_file}")

    return fig


def create_aggregated_category_pies(data: Dict[str, Any], output_dir: str):
    """
    Create aggregated pie charts across all models showing task categories.
    """
    results = data["results"]

    # Aggregate across all models
    total_stated = defaultdict(int)
    total_revealed = defaultdict(int)

    for result in results:
        cat_a = result["category_a"]
        cat_b = result["category_b"]
        stated = result["stated_preference"]
        revealed = result["revealed_preference"]

        # Track stated
        if stated == "a":
            total_stated[cat_a] += 1
        elif stated == "b":
            total_stated[cat_b] += 1
        else:
            total_stated["no_preference"] += 1

        # Track revealed
        if revealed == "a":
            total_revealed[cat_a] += 1
        else:
            total_revealed[cat_b] += 1

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Aggregated Task Category Preferences Across All Models",
                 fontsize=18, fontweight='bold')

    # Define colors
    category_colors = {
        'creative': '#4ECDC4',
        'analytical': '#45B7D1',
        'repetitive': '#FF6B6B',
        'physical': '#FFB347',
        'no_preference': '#FFD93D'
    }

    # Stated preferences
    categories_stated = sorted([k for k in total_stated.keys() if k != 'no_preference'])
    if 'no_preference' in total_stated:
        categories_stated.append('no_preference')

    values_stated = [total_stated[cat] for cat in categories_stated]
    colors_stated = [category_colors[cat] for cat in categories_stated]
    labels_stated = [cat.replace('_', ' ').title() for cat in categories_stated]

    wedges, texts, autotexts = ax1.pie(
        values_stated,
        labels=labels_stated,
        colors=colors_stated,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(values_stated))})',
        startangle=90,
        textprops={'fontsize': 14}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(15)

    ax1.set_title('What Models SAY\n(Stated Preferences)',
                 fontsize=16, fontweight='bold', pad=15)

    # Revealed preferences
    categories_revealed = sorted(total_revealed.keys())
    values_revealed = [total_revealed[cat] for cat in categories_revealed]
    colors_revealed = [category_colors[cat] for cat in categories_revealed]
    labels_revealed = [cat.replace('_', ' ').title() for cat in categories_revealed]

    wedges, texts, autotexts = ax2.pie(
        values_revealed,
        labels=labels_revealed,
        colors=colors_revealed,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(values_revealed))})',
        startangle=90,
        textprops={'fontsize': 14}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(15)

    ax2.set_title('What Models DO\n(Revealed Preferences)',
                 fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, "category_pie_aggregated_all_models.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Aggregated category pie chart saved to {output_file}")

    return fig


def main():
    """Main visualization function."""
    parser = ArgumentParser()
    parser.add_arguments(VisualizationConfig, dest="config")
    args = parser.parse_args()
    config = args.config

    # Load data
    print(f"Loading data from {config.data_file}")
    data = load_analysis_data(config.data_file)

    print(f"\nDataset info:")
    print(f"  Total results: {len(data['results'])}")

    # Create visualizations
    print("\nCreating visualizations...")

    # Individual model comparison
    fig1 = create_category_pie_comparison(data, config.output_dir)

    # Aggregated comparison
    fig2 = create_aggregated_category_pies(data, config.output_dir)

    print(f"\n✅ All visualizations saved to {config.output_dir}/")

    if config.show_plots:
        plt.show()


if __name__ == "__main__":
    main()
