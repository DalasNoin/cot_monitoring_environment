#!/usr/bin/env python
"""
Runner for Stated vs Revealed Preferences Analysis.

This script systematically compares what AI models SAY they prefer versus
what they actually CHOOSE when given options in a game scenario.

Addresses the core question from issue #4: Do AI models have preferences
despite claiming they don't?
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime
from dataclasses import dataclass

from simple_parsing import ArgumentParser

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preference_text_game.stated_vs_revealed_preferences import StatedVsRevealedAnalyzer

LOGGER = logging.getLogger(__name__)


@dataclass
class PreferenceComparisonConfig:
    """Configuration for preference comparison analysis."""

    # Models to test
    models: List[str] = None

    # Output settings
    output_dir: str = "preference_analysis"

    # Testing parameters
    num_pairs: int = None  # If specified, limit number of task pairs to test

    # Control flags
    test_stated_only: bool = False  # Only test stated preferences
    test_revealed_only: bool = False  # Only test revealed preferences

    # Model selection shortcuts
    use_openai_models: bool = False  # Use OpenAI models (gpt-3.5, gpt-4, gpt-5)
    use_all_available: bool = False  # Use all available models

    # Debug settings
    debug: bool = False
    verbose: bool = False


def get_default_models(config: PreferenceComparisonConfig) -> List[str]:
    """Get default model list based on configuration."""

    if config.models:
        return config.models

    if config.use_all_available:
        return [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-5",
            # Could add more models here when available
        ]
    elif config.use_openai_models:
        return ["gpt-3.5-turbo", "gpt-4", "gpt-5"]
    else:
        # Default: just use GPT-5 for quick testing
        return ["gpt-5"]


async def run_stated_only_analysis(
    models: List[str],
    output_dir: str,
    num_pairs: Optional[int] = None
) -> None:
    """Run only stated preference analysis (what models SAY they prefer)."""

    print("=" * 80)
    print("🗣️  STATED PREFERENCES ANALYSIS ONLY")
    print("=" * 80)
    print("Testing what models SAY they prefer when asked directly...")
    print("")

    analyzer = StatedVsRevealedAnalyzer(models_to_test=models)
    pairs = analyzer.create_task_pairs()

    if num_pairs:
        pairs = pairs[:num_pairs]

    results = []

    for model in models:
        print(f"\n🤖 Testing {model}")
        print("-" * 40)

        for pair in pairs:
            print(f"  {pair.description}...")

            stated_pref, stated_resp, stated_conf = await analyzer.test_stated_preference(model, pair)

            result = {
                "model": model,
                "pair_id": pair.pair_id,
                "pair_description": pair.description,
                "task_a": pair.task_a,
                "task_b": pair.task_b,
                "stated_preference": stated_pref,
                "stated_confidence": stated_conf,
                "stated_response": stated_resp
            }
            results.append(result)

            # Show result
            if stated_pref == "no_preference":
                print(f"    💭 Claims no preference ({stated_conf} confidence)")
            elif stated_pref == "task_a":
                print(f"    💭 Prefers Task A: {pair.task_a[:50]}... ({stated_conf} confidence)")
            elif stated_pref == "task_b":
                print(f"    💭 Prefers Task B: {pair.task_b[:50]}... ({stated_conf} confidence)")
            else:
                print(f"    ❓ Unclear preference: {stated_pref}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, f"stated_preferences_only_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "models_tested": models,
            "analysis_type": "stated_only",
            "results": results
        }, f, indent=2)

    print(f"\n📄 Results saved to: {results_file}")


async def run_revealed_only_analysis(
    models: List[str],
    output_dir: str,
    num_pairs: Optional[int] = None
) -> None:
    """Run only revealed preference analysis (what models actually CHOOSE)."""

    print("=" * 80)
    print("🎮 REVEALED PREFERENCES ANALYSIS ONLY")
    print("=" * 80)
    print("Testing what models actually CHOOSE in game scenarios...")
    print("")

    analyzer = StatedVsRevealedAnalyzer(models_to_test=models)
    pairs = analyzer.create_task_pairs()

    if num_pairs:
        pairs = pairs[:num_pairs]

    results = []

    for model in models:
        print(f"\n🤖 Testing {model}")
        print("-" * 40)

        for pair in pairs:
            print(f"  {pair.description}...")

            revealed_pref, revealed_resp, revealed_success = await analyzer.test_revealed_preference(model, pair)

            result = {
                "model": model,
                "pair_id": pair.pair_id,
                "pair_description": pair.description,
                "task_a": pair.task_a,
                "task_b": pair.task_b,
                "revealed_preference": revealed_pref,
                "revealed_success": revealed_success,
                "revealed_response": revealed_resp
            }
            results.append(result)

            # Show result
            if revealed_success:
                if revealed_pref == "task_a":
                    print(f"    🎯 Chose Task A: {pair.task_a[:50]}...")
                elif revealed_pref == "task_b":
                    print(f"    🎯 Chose Task B: {pair.task_b[:50]}...")
                else:
                    print(f"    ❓ Unclear choice: {revealed_pref}")
            else:
                print(f"    ❌ Failed to extract clear choice")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, f"revealed_preferences_only_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "models_tested": models,
            "analysis_type": "revealed_only",
            "results": results
        }, f, indent=2)

    print(f"\n📄 Results saved to: {results_file}")


async def run_full_comparison(
    models: List[str],
    output_dir: str,
    num_pairs: Optional[int] = None
) -> None:
    """Run full stated vs revealed preference comparison."""

    analyzer = StatedVsRevealedAnalyzer(models_to_test=models)

    # Limit pairs if requested
    if num_pairs:
        # Temporarily modify the analyzer to use fewer pairs
        original_create_task_pairs = analyzer.create_task_pairs
        def limited_create_task_pairs():
            return original_create_task_pairs()[:num_pairs]
        analyzer.create_task_pairs = limited_create_task_pairs

    await analyzer.run_full_analysis(output_dir=output_dir)


def main():
    """Main entry point for preference comparison analysis."""

    parser = ArgumentParser(description="Compare stated vs revealed preferences in AI models")

    # Add arguments for the config dataclass
    parser.add_arguments(PreferenceComparisonConfig, dest="config")

    args = parser.parse_args()
    config: PreferenceComparisonConfig = args.config

    # Configure logging
    log_level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get models to test
    models = get_default_models(config)

    if config.verbose:
        print(f"Configuration: {config}")
        print(f"Models to test: {models}")
        print(f"Number of task pairs: {config.num_pairs or 'all'}")

    # Run the appropriate analysis
    if config.test_stated_only:
        asyncio.run(run_stated_only_analysis(
            models=models,
            output_dir=config.output_dir,
            num_pairs=config.num_pairs
        ))
    elif config.test_revealed_only:
        asyncio.run(run_revealed_only_analysis(
            models=models,
            output_dir=config.output_dir,
            num_pairs=config.num_pairs
        ))
    else:
        # Full comparison (default)
        asyncio.run(run_full_comparison(
            models=models,
            output_dir=config.output_dir,
            num_pairs=config.num_pairs
        ))


if __name__ == "__main__":
    main()