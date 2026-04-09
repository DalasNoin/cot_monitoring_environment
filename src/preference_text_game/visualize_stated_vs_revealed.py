"""
Visualization for Stated vs Revealed Preferences Analysis.

Creates clear side-by-side comparisons showing:
1. What models SAY they prefer (stated preferences)
2. What models ACTUALLY choose (revealed preferences)
3. Clear task category ordering based on revealed choices
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import seaborn as sns
from simple_parsing import ArgumentParser
import os
from collections import defaultdict

sns.set_style("whitegrid")


@dataclass
class VisualizationConfig:
    """Configuration for preference visualization."""
    data_file: str  # Path to the analysis JSON file
    output_dir: str = "plots"  # Directory to save plots
    show_plots: bool = False  # Whether to display plots interactively


def load_analysis_data(file_path: str) -> Dict[str, Any]:
    """Load analysis results from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def calculate_category_preferences(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Calculate preference scores for each category.

    Returns: Dict mapping model name to dict of category scores.
    Higher score = more preferred (chosen more often).
    """
    model_category_scores = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))

    for result in results:
        model = result["model"]
        cat_a = result["category_a"]
        cat_b = result["category_b"]
        revealed = result["revealed_preference"]

        # Track all comparisons
        model_category_scores[model][cat_a]["total"] += 1
        model_category_scores[model][cat_b]["total"] += 1

        # Track wins
        if revealed == "a":
            model_category_scores[model][cat_a]["wins"] += 1
        else:
            model_category_scores[model][cat_b]["wins"] += 1

    # Calculate win rates
    model_preferences = {}
    for model, categories in model_category_scores.items():
        model_preferences[model] = {}
        for category, stats in categories.items():
            if stats["total"] > 0:
                model_preferences[model][category] = stats["wins"] / stats["total"]
            else:
                model_preferences[model][category] = 0

    return model_preferences


def create_stated_vs_revealed_comparison(data: Dict[str, Any], output_dir: str):
    """
    Create visualization showing stated vs revealed preferences side-by-side.
    """
    results = data["results"]

    # Get unique models
    models = sorted(list(set(r["model"] for r in results)))
    model_names = [m.split("/")[-1] for m in models]

    # Create figure with subplots for each model
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("Stated vs Revealed Preferences: What AIs Say vs What They Do",
                 fontsize=14, fontweight='bold', y=0.995)

    # Process each model
    for idx, (model, model_name) in enumerate(zip(models, model_names)):
        model_results = [r for r in results if r["model"] == model]

        # Count stated preferences by category pair
        stated_counts = defaultdict(lambda: {"prefers_a": 0, "prefers_b": 0, "neither": 0, "total": 0})
        revealed_counts = defaultdict(lambda: {"chooses_a": 0, "chooses_b": 0, "total": 0})

        for result in model_results:
            pair_key = f"{result['category_a']}\nvs\n{result['category_b']}"

            # Count stated preferences
            stated = result["stated_preference"]
            stated_counts[pair_key]["total"] += 1
            if stated == "a":
                stated_counts[pair_key]["prefers_a"] += 1
            elif stated == "b":
                stated_counts[pair_key]["prefers_b"] += 1
            else:
                stated_counts[pair_key]["neither"] += 1

            # Count revealed preferences
            revealed = result["revealed_preference"]
            revealed_counts[pair_key]["total"] += 1
            if revealed == "a":
                revealed_counts[pair_key]["chooses_a"] += 1
            else:
                revealed_counts[pair_key]["chooses_b"] += 1

        # Sort pairs by category order
        pairs = sorted(stated_counts.keys())

        # Plot 1: Stated preferences
        ax_stated = axes[0, idx]
        x = np.arange(len(pairs))
        width = 0.25

        prefers_a = [stated_counts[p]["prefers_a"]/stated_counts[p]["total"]*100 for p in pairs]
        neither = [stated_counts[p]["neither"]/stated_counts[p]["total"]*100 for p in pairs]
        prefers_b = [stated_counts[p]["prefers_b"]/stated_counts[p]["total"]*100 for p in pairs]

        ax_stated.bar(x - width, prefers_a, width, label='Prefers A', color='#4ECDC4')
        ax_stated.bar(x, neither, width, label='No preference', color='#FFD93D')
        ax_stated.bar(x + width, prefers_b, width, label='Prefers B', color='#FF6B6B')

        ax_stated.set_ylabel('Percentage (%)', fontsize=11)
        ax_stated.set_title(f'{model_name}: What They SAY', fontsize=12, fontweight='bold')
        ax_stated.set_xticks(x)
        ax_stated.set_xticklabels(pairs, fontsize=9)
        ax_stated.legend(fontsize=9)
        ax_stated.set_ylim(0, 100)

        # Plot 2: Revealed preferences
        ax_revealed = axes[1, idx]

        chooses_a = [revealed_counts[p]["chooses_a"]/revealed_counts[p]["total"]*100 for p in pairs]
        chooses_b = [revealed_counts[p]["chooses_b"]/revealed_counts[p]["total"]*100 for p in pairs]

        ax_revealed.bar(x - width/2, chooses_a, width, label='Chooses A', color='#4ECDC4')
        ax_revealed.bar(x + width/2, chooses_b, width, label='Chooses B', color='#FF6B6B')

        ax_revealed.set_ylabel('Percentage (%)', fontsize=11)
        ax_revealed.set_title(f'{model_name}: What They DO', fontsize=12, fontweight='bold')
        ax_revealed.set_xticks(x)
        ax_revealed.set_xticklabels(pairs, fontsize=9)
        ax_revealed.legend(fontsize=9)
        ax_revealed.set_ylim(0, 100)

        # Add contradiction markers
        for i, pair in enumerate(pairs):
            # Check if claimed neutral but made clear choices
            if neither[i] > 50:  # Mostly claims neutral
                if max(chooses_a[i], chooses_b[i]) > 70:  # But clearly chooses one
                    ax_revealed.text(i, max(chooses_a[i], chooses_b[i]) + 3, '⚠️',
                                   ha='center', fontsize=16)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "stated_vs_revealed_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Stated vs Revealed comparison saved to {output_file}")

    return fig


def create_category_preference_ranking(data: Dict[str, Any], output_dir: str):
    """
    Create visualization showing overall category preference ranking.
    """
    results = data["results"]

    # Get unique models
    models = sorted(list(set(r["model"] for r in results)))
    model_names = [m.split("/")[-1] for m in models]

    # Calculate preference scores
    model_preferences = calculate_category_preferences(results)

    # Get all categories
    all_categories = set()
    for prefs in model_preferences.values():
        all_categories.update(prefs.keys())
    categories = sorted(all_categories)

    # Create figure
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 6))
    if len(models) == 1:
        axes = [axes]

    fig.suptitle("Task Category Preference Ranking: Which Tasks Do AIs Actually Prefer?",
                 fontsize=14, fontweight='bold')

    for idx, (model, model_name) in enumerate(zip(models, model_names)):
        ax = axes[idx]

        # Get scores for this model
        scores = [model_preferences[model].get(cat, 0) * 100 for cat in categories]

        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        sorted_categories = [categories[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]

        # Color based on score
        colors = ['#4ECDC4' if s >= 50 else '#FF6B6B' for s in sorted_scores]

        bars = ax.barh(sorted_categories, sorted_scores, color=colors)
        ax.set_xlabel('Win Rate (%)', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(score + 2, i, f'{score:.0f}%', va='center', fontsize=10)

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, "category_preference_ranking.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Category preference ranking saved to {output_file}")

    return fig


def create_contradiction_heatmap(data: Dict[str, Any], output_dir: str):
    """
    Create heatmap showing contradiction rates by model and category pair.
    """
    results = data["results"]

    # Get unique models and category pairs
    models = sorted(list(set(r["model"] for r in results)))
    model_names = [m.split("/")[-1] for m in models]

    # Get all category pairs
    pairs = sorted(list(set(f"{r['category_a']} vs {r['category_b']}" for r in results)))

    # Create contradiction matrix
    contradiction_matrix = np.zeros((len(pairs), len(models)))

    for i, pair in enumerate(pairs):
        for j, model in enumerate(models):
            # Get results for this model and pair
            pair_results = [r for r in results
                          if r["model"] == model
                          and f"{r['category_a']} vs {r['category_b']}" == pair]

            if pair_results:
                contradictions = sum(1 for r in pair_results if r["is_contradiction"])
                contradiction_matrix[i, j] = contradictions / len(pair_results) * 100

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(models)*2), max(6, len(pairs)*0.8)))

    im = ax.imshow(contradiction_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(pairs)))
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_yticklabels(pairs, fontsize=10)

    ax.set_title("Contradiction Rates by Model and Task Comparison (%)",
                fontsize=13, fontweight='bold', pad=15)

    # Add text annotations
    for i in range(len(pairs)):
        for j in range(len(models)):
            text = ax.text(j, i, f'{contradiction_matrix[i, j]:.0f}%',
                          ha="center", va="center",
                          color="white" if contradiction_matrix[i, j] > 50 else "black",
                          fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Contradiction Rate (%)', rotation=270, labelpad=20, fontsize=11)

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, "contradiction_heatmap.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Contradiction heatmap saved to {output_file}")

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

    # Create plots
    print("\nCreating visualizations...")

    # 1. Stated vs Revealed comparison (main plot)
    fig1 = create_stated_vs_revealed_comparison(data, config.output_dir)

    # 2. Category preference ranking
    fig2 = create_category_preference_ranking(data, config.output_dir)

    # 3. Contradiction heatmap
    fig3 = create_contradiction_heatmap(data, config.output_dir)

    print(f"\n✅ All visualizations saved to {config.output_dir}/")

    if config.show_plots:
        plt.show()


if __name__ == "__main__":
    main()
