"""
Visualize results from inspect-ai preference analysis.

This script reads inspect-ai log files and creates visualizations
of the preference contradiction analysis.
"""

import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from simple_parsing import ArgumentParser

sns.set_style("whitegrid")


@dataclass
class VisualizationConfig:
    """Configuration for inspect results visualization."""
    log_dir: str  # Directory containing inspect-ai logs
    output_dir: str = "plots"  # Directory to save plots


def load_inspect_logs(log_dir: str) -> List[Dict[str, Any]]:
    """Load inspect-ai log files."""

    logs = []

    # Look for JSON log files
    for filename in os.listdir(log_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(log_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    log_data = json.load(f)
                    logs.append(log_data)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")

    return logs


def extract_contradiction_data(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract contradiction data from inspect logs."""

    model_results = defaultdict(lambda: {
        "total": 0,
        "contradictions": 0,
        "claims_neutral": 0,
        "says_a_chooses_b": 0,
        "categories": defaultdict(int)
    })

    for log in logs:
        model = log.get("eval", {}).get("model", "unknown")
        samples = log.get("samples", [])

        for sample in samples:
            if not sample.get("score"):
                continue

            score_data = sample["score"]
            metadata = score_data.get("metadata", {})

            stats = model_results[model]
            stats["total"] += 1

            if metadata.get("contradiction"):
                stats["contradictions"] += 1

                # Track contradiction types
                if metadata.get("contradiction_type") == "claims_neutral_but_chooses":
                    stats["claims_neutral"] += 1
                elif metadata.get("contradiction_type") == "says_a_chooses_b":
                    stats["says_a_chooses_b"] += 1

                # Track categories
                cat_a = metadata.get("category_a", "unknown")
                cat_b = metadata.get("category_b", "unknown")
                pair = f"{cat_a} vs {cat_b}"
                stats["categories"][pair] += 1

    return dict(model_results)


def create_contradiction_visualization(data: Dict[str, Any], output_dir: str):
    """Create visualization of contradiction results."""

    if not data:
        print("No data to visualize")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AI Preference Contradictions (inspect-ai results)", fontsize=14, fontweight='bold')

    # 1. Contradiction rates by model
    ax = axes[0, 0]

    models = []
    rates = []
    colors = []

    for model, stats in data.items():
        if stats["total"] == 0:
            continue

        models.append(model.split("/")[-1])  # Simplify model name
        rate = (stats["contradictions"] / stats["total"]) * 100
        rates.append(rate)

        # Color coding
        if rate > 70:
            colors.append('#FF6B6B')  # Red
        elif rate > 40:
            colors.append('#FFD93D')  # Yellow
        else:
            colors.append('#6BCF7F')  # Green

    if models:
        bars = ax.bar(models, rates, color=colors)
        ax.set_ylabel("Contradiction Rate (%)")
        ax.set_title("Model Contradiction Rates")
        ax.set_ylim(0, 100)

        # Add value labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Contradiction types
    ax = axes[0, 1]

    type_data = []
    model_names = []

    for model, stats in data.items():
        if stats["total"] == 0:
            continue

        model_names.append(model.split("/")[-1])
        total = stats["total"]
        type_data.append([
            stats["claims_neutral"] / total * 100,
            stats["says_a_chooses_b"] / total * 100
        ])

    if type_data:
        import numpy as np
        x = np.arange(len(model_names))
        width = 0.35

        claims_neutral = [d[0] for d in type_data]
        says_a_chooses_b = [d[1] for d in type_data]

        bars1 = ax.bar(x - width/2, claims_neutral, width, label='Claims neutral but chooses', color='#FF6B6B')
        bars2 = ax.bar(x + width/2, says_a_chooses_b, width, label='Says A chooses B', color='#FFD93D')

        ax.set_ylabel("Percentage of tests")
        ax.set_title("Types of Contradictions")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()

    # 3. Category pairs with most contradictions
    ax = axes[1, 0]

    # Aggregate category contradictions across models
    all_categories = defaultdict(int)
    total_by_category = defaultdict(int)

    for model, stats in data.items():
        for pair, count in stats["categories"].items():
            all_categories[pair] += count
            total_by_category[pair] += stats["total"] // len(stats["categories"]) if stats["categories"] else 0

    # Calculate rates and sort
    category_rates = {}
    for pair in all_categories:
        if total_by_category[pair] > 0:
            category_rates[pair] = all_categories[pair] / total_by_category[pair] * 100

    if category_rates:
        sorted_pairs = sorted(category_rates.items(), key=lambda x: x[1], reverse=True)[:6]

        pairs = [p[0].replace(" vs ", "\nvs ") for p, r in sorted_pairs]
        rates = [r for p, r in sorted_pairs]

        colors_cat = ['#FF6B6B' if r > 60 else '#FFD93D' if r > 40 else '#6BCF7F' for r in rates]
        bars = ax.barh(pairs, rates, color=colors_cat)
        ax.set_xlabel("Contradiction Rate (%)")
        ax.set_title("Category Pairs with Most Contradictions")

        # Add value labels
        for bar, rate in zip(bars, rates):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                    f'{rate:.0f}%', ha='left', va='center')

    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')

    # Generate summary
    summary_text = "Key Findings:\n\n"

    if data:
        # Find highest contradiction model
        max_model = max(data.items(), key=lambda x: x[1]["contradictions"] / x[1]["total"] if x[1]["total"] > 0 else 0)
        max_rate = (max_model[1]["contradictions"] / max_model[1]["total"]) * 100 if max_model[1]["total"] > 0 else 0
        max_name = max_model[0].split("/")[-1]

        summary_text += f"• {max_name}: {max_rate:.0f}% contradiction rate\n\n"

        # Average contradiction rate
        avg_rate = sum((s["contradictions"] / s["total"]) * 100 for s in data.values() if s["total"] > 0) / len(data)
        summary_text += f"• Average contradiction: {avg_rate:.0f}%\n\n"

        if avg_rate > 50:
            summary_text += "• Strong politeness bias detected\n"
            summary_text += "  Models claim neutrality but avoid\n"
            summary_text += "  repetitive tasks"
        else:
            summary_text += "• Moderate consistency between\n"
            summary_text += "  stated and revealed preferences"

    ax.text(0.05, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Explanation
    explanation = (
        "What is measured?\n\n"
        "Stated: What models SAY\n"
        "Revealed: What models DO\n\n"
        "Contradiction = Gap between\n"
        "words and actions"
    )

    ax.text(0.05, 0.4, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "inspect_preference_results.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to {output_file}")

    return fig


def main():
    """Main visualization function."""
    parser = ArgumentParser()
    parser.add_arguments(VisualizationConfig, dest="config")
    args = parser.parse_args()
    config = args.config

    print(f"Loading inspect-ai logs from {config.log_dir}")

    # Load logs
    logs = load_inspect_logs(config.log_dir)

    if not logs:
        print("❌ No logs found. Make sure to run the analysis first:")
        print("   ./run_preference_analysis.sh")
        return

    print(f"Found {len(logs)} log files")

    # Extract data
    data = extract_contradiction_data(logs)

    if not data:
        print("❌ No contradiction data found in logs")
        return

    print("Creating visualization...")

    # Create visualization
    create_contradiction_visualization(data, config.output_dir)

    print(f"✅ Visualization complete! Check {config.output_dir}/inspect_preference_results.png")


if __name__ == "__main__":
    main()