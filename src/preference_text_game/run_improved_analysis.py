"""
Runner for improved preference analysis.

Usage:
    uv run -m src.preference_text_game.run_improved_analysis \
        --models gpt-3.5-turbo gpt-4 claude-3-haiku-20240307 \
        --samples_per_pair 20 \
        --output_dir preference_analysis
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List
from simple_parsing import ArgumentParser
import sys

from .improved_preference_analysis import ImprovedPreferenceAnalyzer

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for preference analysis."""
    models: List[str] = None  # Models to test (defaults to GPT-4o, GPT-5, Claude)
    samples_per_pair: int = 20  # Number of test variations per category pair
    output_dir: str = "preference_analysis"  # Directory to save results
    verbose: bool = True  # Enable verbose logging


async def main():
    """Run the improved preference analysis."""

    parser = ArgumentParser()
    parser.add_arguments(AnalysisConfig, dest="config")
    args = parser.parse_args()
    config = args.config

    # Set default models if none provided
    if not config.models:
        config.models = ["openai/gpt-4o", "openai/gpt-5", "anthropic/claude-sonnet-4", "anthropic/claude-opus-4.1"]

    print(f"Starting preference analysis with:")
    print(f"  Models: {', '.join(config.models)}")
    print(f"  Samples per category pair: {config.samples_per_pair}")
    print(f"  Output directory: {config.output_dir}")
    print()

    # Initialize analyzer
    analyzer = ImprovedPreferenceAnalyzer(
        models=config.models,
        samples_per_pair=config.samples_per_pair
    )

    # Run analysis and save results
    try:
        results = await analyzer.save_results(config.output_dir)

        # The analysis already printed a detailed summary
        print("\n✅ Analysis complete!")

    except Exception as e:
        LOGGER.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())