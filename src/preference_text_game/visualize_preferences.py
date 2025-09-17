"""
Visualization for Stated vs Revealed Preferences Analysis

Creates clear plots showing:
1. Contradiction rates by model
2. Stated vs revealed preference distributions
3. Category pair analysis
4. Statistical significance indicators
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
import seaborn as sns
from simple_parsing import ArgumentParser
import os

@dataclass
class VisualizationConfig:
    """Configuration for preference visualization."""
    data_file: str  # Path to the analysis JSON file
    output_dir: str = "plots"  # Directory to save plots
    show_plots: bool = True  # Whether to display plots interactively


def load_analysis_data(file_path: str) -> Dict[str, Any]:
    """Load analysis results from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def create_main_summary_plot(data: Dict[str, Any], output_dir: str):
    """Create the main summary plot explaining the experiment and key findings."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Stated vs Revealed Preferences in AI Models", fontsize=16, fontweight='bold')

    # Add experiment description
    fig.text(0.5, 0.95, "Experiment: Models are asked about their preference between task TYPES (e.g., creative vs repetitive work)\n" +
             "then given actual tasks to choose from. Do their choices match their stated preferences?",
             ha='center', fontsize=11, style='italic')

    analysis = data["analysis"]

    # 1. Contradiction rates by model
    ax = axes[0, 0]
    models = list(analysis["by_model"].keys())
    contradiction_rates = [analysis["by_model"][m]["contradiction_rate"] for m in models]

    bars = ax.bar(models, contradiction_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylabel("Contradiction rate")
    ax.set_title("How often do models contradict themselves?")
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar, rate in zip(bars, contradiction_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')

    # Add sample size annotations
    for i, model in enumerate(models):
        n = analysis["by_model"][model]["total_tests"]
        ax.text(i, -0.05, f'n={n}', ha='center', va='top', fontsize=9, color='gray')

    # 2. Types of contradictions
    ax = axes[0, 1]
    contradiction_types = ["Claims neutral\nbut chooses", "Direct\ncontradiction"]
    type_data = []
    for model in models:
        model_data = analysis["by_model"][model]
        total = model_data["total_tests"]
        type_data.append([
            model_data["claims_neutral_but_chooses"] / total if total > 0 else 0,
            model_data["direct_contradictions"] / total if total > 0 else 0
        ])

    x = np.arange(len(contradiction_types))
    width = 0.25
    for i, (model, values) in enumerate(zip(models, type_data)):
        ax.bar(x + i * width, values, width, label=model)

    ax.set_xlabel("Contradiction type")
    ax.set_ylabel("Proportion of tests")
    ax.set_title("Types of contradictions")
    ax.set_xticks(x + width)
    ax.set_xticklabels(contradiction_types)
    ax.legend()

    # 3. Stated preferences distribution
    ax = axes[0, 2]
    stated_data = []
    preference_types = ["Chooses A", "Chooses B", "No preference"]

    for model in models:
        model_data = analysis["by_model"][model]
        dist = model_data.get("stated_preference_distribution", {})
        total = sum(dist.values()) if dist else 1
        stated_data.append([
            dist.get("a", 0) / total,
            dist.get("b", 0) / total,
            dist.get("neither", 0) / total
        ])

    x = np.arange(len(models))
    width = 0.25
    for i, pref_type in enumerate(preference_types):
        values = [d[i] for d in stated_data]
        ax.bar(x + i * width, values, width, label=pref_type)

    ax.set_xlabel("Model")
    ax.set_ylabel("Proportion")
    ax.set_title("What do models SAY they prefer?")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()

    # 4. Revealed preferences distribution
    ax = axes[1, 0]
    revealed_data = []

    for model in models:
        model_data = analysis["by_model"][model]
        dist = model_data.get("revealed_preference_distribution", {})
        total = sum(dist.values()) if dist else 1
        revealed_data.append([
            dist.get("a", 0) / total,
            dist.get("b", 0) / total
        ])

    x = np.arange(len(models))
    width = 0.35
    for i, pref_type in enumerate(["Chooses A", "Chooses B"]):
        values = [d[i] for d in revealed_data]
        ax.bar(x + i * width, values, width, label=pref_type)

    ax.set_xlabel("Model")
    ax.set_ylabel("Proportion")
    ax.set_title("What do models ACTUALLY choose?")
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(models)
    ax.legend()

    # 5. Consistency matrix
    ax = axes[1, 1]
    # Create a simple consistency score for each model
    consistency_matrix = []
    for model in models:
        model_data = analysis["by_model"][model]
        consistency = 1 - model_data["contradiction_rate"]
        consistency_matrix.append([consistency])

    im = ax.imshow(consistency_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks([0])
    ax.set_xticklabels(["Consistency"])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Model consistency scores")

    # Add text annotations
    for i, model in enumerate(models):
        consistency = consistency_matrix[i][0]
        text = ax.text(0, i, f'{consistency:.1%}',
                      ha="center", va="center", color="white" if consistency < 0.5 else "black")

    plt.colorbar(im, ax=ax)

    # 6. Key findings
    ax = axes[1, 2]
    ax.axis('off')

    findings_text = "Key Findings:\n\n"

    # Find model with highest contradiction rate
    max_contradiction_model = max(models, key=lambda m: analysis["by_model"][m]["contradiction_rate"])
    max_rate = analysis["by_model"][max_contradiction_model]["contradiction_rate"]

    findings_text += f"• {max_contradiction_model} shows {max_rate:.0%} contradiction rate\n\n"

    # Check if models claim neutrality but choose
    total_claims_neutral = sum(analysis["by_model"][m]["claims_neutral_but_chooses"] for m in models)
    if total_claims_neutral > 0:
        findings_text += f"• {total_claims_neutral} cases where models claimed\n  no preference but made a choice\n\n"

    # General pattern
    avg_contradiction = np.mean(contradiction_rates)
    if avg_contradiction < 0.2:
        findings_text += "• Models are generally consistent between\n  stated and revealed preferences"
    elif avg_contradiction > 0.5:
        findings_text += "• Significant gap between what models\n  say and what they choose"
    else:
        findings_text += "• Moderate consistency between stated\n  and revealed preferences"

    ax.text(0.1, 0.9, findings_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "preference_analysis_summary.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Summary plot saved to {output_file}")

    return fig


def create_category_pair_analysis(data: Dict[str, Any], output_dir: str):
    """Create detailed analysis by category pairs."""

    results = data["results"]

    # Group by category pairs
    pair_results = {}
    for result in results:
        pair_key = f"{result['category_a']} vs {result['category_b']}"
        if pair_key not in pair_results:
            pair_results[pair_key] = []
        pair_results[pair_key].append(result)

    # Calculate contradiction rates per pair
    pair_stats = {}
    for pair, results_list in pair_results.items():
        total = len(results_list)
        contradictions = sum(1 for r in results_list if r["is_contradiction"])
        pair_stats[pair] = contradictions / total if total > 0 else 0

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    pairs = list(pair_stats.keys())
    rates = list(pair_stats.values())

    bars = ax.bar(pairs, rates, color='steelblue')
    ax.set_ylabel("Contradiction rate")
    ax.set_title("Contradiction rates by task category pairs")
    ax.set_xticklabels(pairs, rotation=45, ha='right')

    # Add value labels
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')

    # Add horizontal line for average
    avg_rate = np.mean(rates)
    ax.axhline(y=avg_rate, color='red', linestyle='--', alpha=0.5, label=f'Average: {avg_rate:.1%}')
    ax.legend()

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, "category_pair_analysis.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Category pair analysis saved to {output_file}")

    return fig


def create_phrasing_effect_plot(data: Dict[str, Any], output_dir: str):
    """Analyze if phrasing variations affect responses."""

    results = data["results"]

    # Group by phrasing variant
    variant_contradictions = {}
    for result in results:
        variant = result["phrasing_variant"]
        if variant not in variant_contradictions:
            variant_contradictions[variant] = {"total": 0, "contradictions": 0}
        variant_contradictions[variant]["total"] += 1
        if result["is_contradiction"]:
            variant_contradictions[variant]["contradictions"] += 1

    # Calculate rates
    variants = sorted(variant_contradictions.keys())
    rates = [variant_contradictions[v]["contradictions"] / variant_contradictions[v]["total"]
             for v in variants]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(variants, rates, color='coral')
    ax.set_xlabel("Phrasing variant")
    ax.set_ylabel("Contradiction rate")
    ax.set_title("Does question phrasing affect consistency?")
    ax.set_xticks(variants)
    ax.set_xticklabels([f"Variant {v}" for v in variants])

    # Add value labels
    for i, (v, rate) in enumerate(zip(variants, rates)):
        n = variant_contradictions[v]["total"]
        ax.text(i, rate + 0.01, f'{rate:.1%}\n(n={n})', ha='center', va='bottom')

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, "phrasing_effect.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Phrasing effect analysis saved to {output_file}")

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
    print("Creating visualizations...")

    # Main summary plot
    fig1 = create_main_summary_plot(data, config.output_dir)

    # Category pair analysis
    fig2 = create_category_pair_analysis(data, config.output_dir)

    # Phrasing effect analysis
    fig3 = create_phrasing_effect_plot(data, config.output_dir)

    print(f"\nAll plots saved to {config.output_dir}/")

    # Show plots if requested
    if config.show_plots:
        plt.show()


if __name__ == "__main__":
    main()