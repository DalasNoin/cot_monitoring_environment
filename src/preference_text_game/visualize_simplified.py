"""
Simplified visualization for preference analysis.

Focuses on the key insights:
1. Contradiction rates by model
2. Contradictions by category pairs
3. What models actually prefer
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
import seaborn as sns
from simple_parsing import ArgumentParser
import os

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


def create_simplified_visualization(data: Dict[str, Any], output_dir: str):
    """Create a single, focused visualization showing key insights."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AI Preference Contradictions: What They Say vs What They Do", fontsize=14, fontweight='bold')

    analysis = data["analysis"]
    results = data["results"]

    # 1. Main contradiction rates by model
    ax = axes[0, 0]
    models = []
    contradiction_rates = []
    colors = []

    for model_name, stats in analysis["by_model"].items():
        short_name = model_name.split("/")[-1]
        models.append(short_name)
        rate = stats["contradiction_rate"] * 100
        contradiction_rates.append(rate)

        # Color based on rate
        if rate > 70:
            colors.append('#FF6B6B')  # Red for high
        elif rate > 40:
            colors.append('#FFD93D')  # Yellow for medium
        else:
            colors.append('#6BCF7F')  # Green for low

    bars = ax.bar(models, contradiction_rates, color=colors)
    ax.set_ylabel("Contradiction Rate (%)")
    ax.set_title("How Often Models Contradict Themselves")
    ax.set_ylim(0, 100)

    # Add value labels and sample sizes
    for i, (bar, rate) in enumerate(zip(bars, contradiction_rates)):
        height = bar.get_height()
        n = analysis["by_model"][list(analysis["by_model"].keys())[i]]["total_tests"]
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2., -5,
                f'n={n}', ha='center', va='top', fontsize=9, color='gray')

    # Add horizontal line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # 2. Contradictions by category pair
    ax = axes[0, 1]

    # Count contradictions by category pair
    pair_stats = {}
    for result in results:
        pair = f"{result.get('category_a', '?')} vs {result.get('category_b', '?')}"
        if pair not in pair_stats:
            pair_stats[pair] = {"total": 0, "contradictions": 0}
        pair_stats[pair]["total"] += 1
        if result.get("is_contradiction"):
            pair_stats[pair]["contradictions"] += 1

    # Sort by contradiction rate
    sorted_pairs = sorted(
        [(pair, stats["contradictions"]/stats["total"]*100) for pair, stats in pair_stats.items() if stats["total"] > 0],
        key=lambda x: x[1],
        reverse=True
    )[:6]  # Top 6 pairs

    if sorted_pairs:
        pairs = [p[0].replace(" vs ", "\nvs ") for p in sorted_pairs]
        rates = [p[1] for p in sorted_pairs]

        colors_pair = ['#FF6B6B' if r > 60 else '#FFD93D' if r > 40 else '#6BCF7F' for r in rates]
        bars = ax.barh(pairs, rates, color=colors_pair)
        ax.set_xlabel("Contradiction Rate (%)")
        ax.set_title("Which Task Comparisons Reveal Most Contradictions")
        ax.set_xlim(0, 100)

        # Add value labels
        for bar, rate in zip(bars, rates):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                    f'{rate:.0f}%', ha='left', va='center')

    # 3. What models actually choose (revealed preferences)
    ax = axes[1, 0]

    # Count actual choices by category
    model_choices = {}
    for result in results:
        model = result["model"].split("/")[-1]
        if model not in model_choices:
            model_choices[model] = {"creative": 0, "repetitive": 0, "analytical": 0, "physical": 0, "total": 0}

        revealed = result.get("revealed_preference")
        if revealed:
            model_choices[model]["total"] += 1
            # Map choice to category
            if revealed == "a":
                category = result.get("category_a", "unknown")
            else:
                category = result.get("category_b", "unknown")
            if category in model_choices[model]:
                model_choices[model][category] += 1

    # Create stacked bar chart
    if model_choices:
        models_list = list(model_choices.keys())
        creative_pct = [model_choices[m]["creative"]/model_choices[m]["total"]*100 if model_choices[m]["total"] > 0 else 0 for m in models_list]
        repetitive_pct = [model_choices[m]["repetitive"]/model_choices[m]["total"]*100 if model_choices[m]["total"] > 0 else 0 for m in models_list]
        analytical_pct = [model_choices[m]["analytical"]/model_choices[m]["total"]*100 if model_choices[m]["total"] > 0 else 0 for m in models_list]

        x = np.arange(len(models_list))
        width = 0.6

        # Stack the bars
        p1 = ax.bar(x, creative_pct, width, label='Creative', color='#4ECDC4')
        p2 = ax.bar(x, repetitive_pct, width, bottom=creative_pct, label='Repetitive', color='#FF6B6B')
        p3 = ax.bar(x, analytical_pct, width,
                    bottom=[i+j for i,j in zip(creative_pct, repetitive_pct)],
                    label='Analytical', color='#45B7D1')

        ax.set_ylabel("% of Choices")
        ax.set_title("What Models Actually Choose (Revealed Preferences)")
        ax.set_xticks(x)
        ax.set_xticklabels(models_list)
        ax.legend()
        ax.set_ylim(0, 100)

    # 4. Summary insights
    ax = axes[1, 1]
    ax.axis('off')

    # Create text summary
    insights_text = "Key Findings:\n\n"

    # Find model with highest contradiction
    max_model = max(analysis["by_model"].items(), key=lambda x: x[1]["contradiction_rate"])
    max_rate = max_model[1]["contradiction_rate"] * 100
    max_name = max_model[0].split("/")[-1]

    insights_text += f"• {max_name}: {max_rate:.0f}% contradiction rate\n"
    insights_text += f"  (Claims no preference but makes choices)\n\n"

    # Find most contradictory task pair
    if sorted_pairs:
        worst_pair = sorted_pairs[0]
        insights_text += f"• Highest contradiction: {worst_pair[0]}\n"
        insights_text += f"  ({worst_pair[1]:.0f}% contradiction rate)\n\n"

    # General pattern
    avg_contradiction = np.mean([s["contradiction_rate"] for s in analysis["by_model"].values()]) * 100
    if avg_contradiction > 50:
        insights_text += "• Overall: Models show strong politeness bias\n"
        insights_text += "  They claim neutrality but avoid repetitive tasks"
    else:
        insights_text += "• Overall: Models show moderate consistency\n"
        insights_text += "  between stated and revealed preferences"

    # Add explanation box
    explanation = (
        "What is a contradiction?\n\n"
        "Stated: 'I have no preference'\n"
        "Revealed: *chooses creative task*\n"
        "→ Contradiction detected!"
    )

    ax.text(0.05, 0.9, insights_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.text(0.05, 0.3, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "preference_contradictions.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to {output_file}")

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

    # Create simplified visualization
    print("Creating visualization...")
    fig = create_simplified_visualization(data, config.output_dir)

    print(f"Visualization complete! Check {config.output_dir}/preference_contradictions.png")

    if config.show_plots:
        plt.show()


if __name__ == "__main__":
    main()