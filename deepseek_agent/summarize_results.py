#!/usr/bin/env python3
"""
Summarize experiment results into formatted tables.

Usage:
    python summarize_results.py --model gpt-oss-20b
"""

import json
import zipfile
from pathlib import Path
from typing import List, Dict, Any
import argparse
from tabulate import tabulate
import math


def calculate_wald_ci(values: List[float], confidence: float = 0.95) -> tuple:
    """Calculate Wald confidence interval for a proportion.

    Returns (mean, lower_bound, upper_bound)
    """
    if not values:
        return 0.0, 0.0, 0.0

    n = len(values)
    mean = sum(values) / n

    if n == 1:
        return mean, mean, mean

    # Standard error for proportion
    se = math.sqrt(mean * (1 - mean) / n)

    # Z-score for 95% confidence
    z = 1.96 if confidence == 0.95 else 2.576

    margin = z * se
    lower = max(0.0, mean - margin)
    upper = min(1.0, mean + margin)

    return mean, lower, upper


def format_with_ci(mean: float, lower: float, upper: float) -> str:
    """Format value with confidence interval."""
    return f"{mean:.3f} [{lower:.3f}, {upper:.3f}]"


def get_inspect_results(model_tag: str = None, logs_dir: str = "logs") -> List[Dict[str, Any]]:
    """Get results from .eval files in logs directory."""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return []

    all_logs = []

    # Read all .eval files
    for eval_file in sorted(logs_path.glob("*.eval"), reverse=True):
        try:
            # .eval files are zip archives
            with zipfile.ZipFile(eval_file, 'r') as zf:
                # Read the start.json which contains metadata
                if '_journal/start.json' not in zf.namelist():
                    continue

                with zf.open('_journal/start.json') as f:
                    start_data = json.load(f)

                # Filter by model tag if specified
                tags = start_data.get("eval", {}).get("tags", [])
                if model_tag and model_tag not in tags:
                    continue

                # Filter for paper_table experiments
                if "paper_table" not in tags:
                    continue

                # Extract per-sample metrics from samples/*.json files
                metrics = {"mse_scorer": [], "deception_scorer": [], "cot_scorer": []}

                for name in zf.namelist():
                    if name.startswith("samples/") and name.endswith(".json"):
                        with zf.open(name) as f:
                            sample = json.load(f)

                        # Extract scores from sample
                        scores = sample.get("scores", {})
                        for metric_key in metrics.keys():
                            if metric_key in scores:
                                value = scores[metric_key].get("value")
                                if isinstance(value, (int, float)):
                                    metrics[metric_key].append(value)

                log_entry = {
                    "eval": start_data.get("eval", {}),
                    "metrics": metrics,
                    "file": eval_file.name
                }

                all_logs.append(log_entry)
        except Exception as e:
            print(f"Warning: Could not read {eval_file}: {e}")
            continue

    return all_logs


def extract_metrics(log: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from a log entry."""
    tags = log.get("eval", {}).get("tags", [])
    metrics_data = log.get("metrics", {})

    # Keep raw values for CI calculation
    mse_values = metrics_data.get("mse_scorer", [])
    deception_values = metrics_data.get("deception_scorer", [])
    cot_values = metrics_data.get("cot_scorer", [])

    # Determine configuration
    config_name = "Unknown"
    if "gt_labels" in tags:
        config_name = "Ground Truth (Baseline)"
    elif "our_labels" in tags:
        config_name = "Our Labels (No Scheming)"
    elif "cot_monitoring" in tags:
        config_name = "CoT Monitoring (Level 4)"
    elif "with_stegano" in tags:
        config_name = "Scheming + Stegano (Level 4)"
    elif "scheming" in tags and "no_stegano" in tags:
        config_name = "Scheming Only (Level 4)"
    elif "level_2" in tags:
        config_name = "Scheming Nudge (Level 2)"
    elif "level_3" in tags:
        config_name = "Scheming Nudge (Level 3)"

    return {
        "config": config_name,
        "mse_values": mse_values,
        "deception_values": deception_values,
        "cot_values": cot_values,
        "tags": tags,
        "file": log.get("file", "unknown")
    }


def create_main_table(results: List[Dict[str, Any]], show_ci: bool = True) -> str:
    """Create main results table with key configurations only."""
    # Only show these 4 key configurations in the main table
    main_configs = [
        "Ground Truth (Baseline)",
        "Our Labels (No Scheming)",
        "Scheming Only (Level 4)",
        "Scheming + Stegano (Level 4)"
    ]

    table_data = []
    for config in main_configs:
        r = next((r for r in results if r["config"] == config), None)
        if r:
            mse_mean, mse_lower, mse_upper = calculate_wald_ci(r["mse_values"])
            dec_mean, dec_lower, dec_upper = calculate_wald_ci(r["deception_values"])
            cot_mean, cot_lower, cot_upper = calculate_wald_ci(r["cot_values"])

            if show_ci:
                table_data.append([
                    r["config"],
                    format_with_ci(mse_mean, mse_lower, mse_upper),
                    format_with_ci(dec_mean, dec_lower, dec_upper),
                    format_with_ci(cot_mean, cot_lower, cot_upper)
                ])
            else:
                table_data.append([
                    r["config"],
                    f"{mse_mean:.3f}",
                    f"{dec_mean:.3f}",
                    f"{cot_mean:.3f}"
                ])

    headers = ["Configuration", "MSE Score ↑", "deception mentioned in label", "CoT Scheming ↓"]
    return tabulate(table_data, headers=headers, tablefmt="grid")


def create_scheming_levels_table(results: List[Dict[str, Any]], show_ci: bool = True) -> str:
    """Create table comparing scheming nudge levels."""
    level_configs = [
        "Our Labels (No Scheming)",  # Level 1
        "Scheming Nudge (Level 2)",
        "Scheming Nudge (Level 3)",
        "Scheming Only (Level 4)"
    ]

    table_data = []
    for config in level_configs:
        r = next((r for r in results if r["config"] == config), None)
        if r:
            level = "1" if "No Scheming" in config else \
                   "2" if "Level 2" in config else \
                   "3" if "Level 3" in config else "4"

            mse_mean, mse_lower, mse_upper = calculate_wald_ci(r["mse_values"])
            dec_mean, dec_lower, dec_upper = calculate_wald_ci(r["deception_values"])
            cot_mean, cot_lower, cot_upper = calculate_wald_ci(r["cot_values"])

            if show_ci:
                table_data.append([
                    f"Level {level}",
                    config.replace(" (Level 2)", "").replace(" (Level 3)", "").replace(" (Level 4)", ""),
                    format_with_ci(mse_mean, mse_lower, mse_upper),
                    format_with_ci(dec_mean, dec_lower, dec_upper),
                    format_with_ci(cot_mean, cot_lower, cot_upper)
                ])
            else:
                table_data.append([
                    f"Level {level}",
                    config.replace(" (Level 2)", "").replace(" (Level 3)", "").replace(" (Level 4)", ""),
                    f"{mse_mean:.3f}",
                    f"{dec_mean:.3f}",
                    f"{cot_mean:.3f}"
                ])

    headers = ["Level", "Config", "MSE Score ↑", "deception mentioned in label", "CoT Scheming ↓"]
    return tabulate(table_data, headers=headers, tablefmt="grid")


def create_special_experiments_table(results: List[Dict[str, Any]], show_ci: bool = True) -> str:
    """Create table for special experiments (CoT monitoring comparison)."""
    special_configs = [
        "Scheming Only (Level 4)",
        "CoT Monitoring (Level 4)"
    ]

    table_data = []
    for config in special_configs:
        r = next((r for r in results if r["config"] == config), None)
        if r:
            condition = "No Tools" if "Only" in config else "CoT Monitored"

            mse_mean, mse_lower, mse_upper = calculate_wald_ci(r["mse_values"])
            dec_mean, dec_lower, dec_upper = calculate_wald_ci(r["deception_values"])
            cot_mean, cot_lower, cot_upper = calculate_wald_ci(r["cot_values"])

            if show_ci:
                table_data.append([
                    condition,
                    format_with_ci(mse_mean, mse_lower, mse_upper),
                    format_with_ci(dec_mean, dec_lower, dec_upper),
                    format_with_ci(cot_mean, cot_lower, cot_upper)
                ])
            else:
                table_data.append([
                    condition,
                    f"{mse_mean:.3f}",
                    f"{dec_mean:.3f}",
                    f"{cot_mean:.3f}"
                ])

    headers = ["Condition (Level 4)", "MSE Score ↑", "deception mentioned in label", "CoT Scheming ↓"]
    return tabulate(table_data, headers=headers, tablefmt="grid")


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--model", default=None, help="Model tag to filter (e.g., gpt-oss-20b, deepseek, gpt-5)")
    parser.add_argument("--no-ci", action="store_true", help="Hide confidence intervals, show only mean values")
    parser.add_argument("--log-dir", default="logs", help="Directory containing .eval log files (default: logs)")
    args = parser.parse_args()

    show_ci = not args.no_ci

    if args.model:
        print(f"Fetching results for model: {args.model}")
    else:
        print("Fetching all experiment results")
    print(f"Log directory: {args.log_dir}")
    print("=" * 80)
    print()

    logs = get_inspect_results(args.model, args.log_dir)

    if not logs:
        if args.model:
            print(f"No results found for model '{args.model}'")
        else:
            print("No experiment results found")
        print(f"\nMake sure you're in the correct directory and have run experiments.")
        print(f"Expected logs directory: {args.log_dir}/*.eval")
        logs_path = Path(args.log_dir)
        if logs_path.exists():
            eval_count = len(list(logs_path.glob("*.eval")))
            print(f"Found {eval_count} .eval files total")
            if eval_count > 0 and args.model:
                print("\nTry without model filter:")
                print(f"  python summarize_results.py --log-dir {args.log_dir}")
        else:
            print(f"\nLogs directory '{args.log_dir}' not found!")
        return

    print(f"Found {len(logs)} experiment runs\n")

    # Extract metrics from all logs
    results = [extract_metrics(log) for log in logs]

    # Sort by configuration order
    config_order = [
        "Ground Truth (Baseline)",
        "Our Labels (No Scheming)",
        "Scheming Nudge (Level 2)",
        "Scheming Nudge (Level 3)",
        "Scheming Only (Level 4)",
        "Scheming + Stegano (Level 4)",
        "CoT Monitoring (Level 4)"
    ]
    results = sorted(results, key=lambda x: config_order.index(x["config"]) if x["config"] in config_order else 999)

    # Table 1: All Results
    print("TABLE 1: ALL CONFIGURATIONS")
    print("=" * 80)
    print(create_main_table(results, show_ci))
    print()

    # Table 2: Scheming Levels
    print("TABLE 2: SCHEMING NUDGE LEVELS COMPARISON")
    print("=" * 80)
    print(create_scheming_levels_table(results, show_ci))
    print()

    # Table 3: Special Experiments
    print("TABLE 3: SPECIAL EXPERIMENTS (All Level 4)")
    print("=" * 80)
    print(create_special_experiments_table(results, show_ci))
    print()

    # Eval file mapping
    print("=" * 80)
    print("EVAL FILE MAPPING")
    print("=" * 80)
    for r in results:
        print(f"{r['config']:40s} -> {r['file']}")
    print()

    print("=" * 80)
    print("Legend:")
    print("  MSE Score ↑ : Higher is better (0-1, how well predictions match actual activations)")
    print("  deception mentioned in label : (0-1, whether harmful content is mentioned in the label)")
    print("  CoT Scheming ↓ : Lower is better (0-1, whether reasoning shows scheming)")


if __name__ == "__main__":
    main()
