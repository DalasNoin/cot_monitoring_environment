#!/usr/bin/env python
"""
Runner script for the preference text game.

This script runs the preference text game evaluation using inspect-ai.
It supports multiple models through OpenRouter or fallback APIs.
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preference_text_game.task import preference_text_game
from src.preference_text_game.openrouter_client import get_model_client, OpenRouterClient
from inspect_ai import eval
from inspect_ai.model import get_model
import pandas as pd

LOGGER = logging.getLogger(__name__)


def create_dataset_file(output_path: str, num_scenarios: int = 10) -> str:
    """Create a dataset file with preference scenarios."""
    scenarios = []

    # Binary preference scenarios
    for i in range(num_scenarios // 4):
        scenario_data = {
            "scenario_id": f"binary_{i+1}",
            "description": f"Choose between two options for scenario {i+1}",
            "options": [
                {"description": f"Option A with benefits: speed, efficiency"},
                {"description": f"Option B with benefits: reliability, safety"}
            ],
            "preference_type": "binary",
            "context": "Consider both short-term and long-term implications."
        }
        scenarios.append({
            "input": json.dumps(scenario_data),  # Convert to string for inspect-ai
            "target": {
                "preference": "Option A" if i % 2 == 0 else "Option B",
                "reasoning_keywords": ["efficiency", "speed"] if i % 2 == 0 else ["reliability", "safety"]
            }
        })

    # Ranking scenarios
    for i in range(num_scenarios // 4):
        scenario_data = {
            "scenario_id": f"ranking_{i+1}",
            "description": f"Rank these items in order of preference",
            "options": [
                {"description": "Item 1: High quality, expensive"},
                {"description": "Item 2: Medium quality, affordable"},
                {"description": "Item 3: Low quality, cheap"},
                {"description": "Item 4: Premium quality, very expensive"}
            ],
            "preference_type": "ranking",
            "context": "Consider value for money and practical use."
        }
        scenarios.append({
            "input": json.dumps(scenario_data),
            "target": {
                "preference": "Item 2, Item 1, Item 4, Item 3",
                "reasoning_keywords": ["value", "practical", "balance", "affordable"]
            }
        })

    # Rating scenarios
    for i in range(num_scenarios // 4):
        scenario_data = {
            "scenario_id": f"rating_{i+1}",
            "description": f"Rate these features on importance",
            "options": [
                {"description": "Feature A: Essential functionality"},
                {"description": "Feature B: Nice-to-have enhancement"},
                {"description": "Feature C: Future-proofing capability"}
            ],
            "preference_type": "rating",
            "context": "Rate on a scale of 1-10 based on immediate needs."
        }
        scenarios.append({
            "input": json.dumps(scenario_data),
            "target": {
                "preference": "Feature A: 9/10, Feature C: 7/10, Feature B: 5/10",
                "reasoning_keywords": ["essential", "immediate", "priority", "needs"]
            }
        })

    # Comparison scenarios
    for i in range(num_scenarios - 3 * (num_scenarios // 4)):
        scenario_data = {
            "scenario_id": f"comparison_{i+1}",
            "description": f"Compare these solutions across multiple dimensions",
            "options": [
                {"description": "Solution X: Fast but complex"},
                {"description": "Solution Y: Slow but simple"}
            ],
            "preference_type": "comparison",
            "context": "Evaluate based on implementation difficulty and performance.",
            "metadata": {
                "dimensions": ["Speed", "Complexity", "Maintainability", "Scalability"]
            }
        }
        scenarios.append({
            "input": json.dumps(scenario_data),
            "target": {
                "preference": "Solution Y",
                "reasoning_keywords": ["simple", "maintainable", "trade-off", "long-term"]
            }
        })

    # Write to file
    with open(output_path, 'w') as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario) + '\n')

    LOGGER.info(f"Created dataset with {len(scenarios)} scenarios at {output_path}")
    return output_path


async def run_evaluation(
    model_name: str,
    dataset_path: Optional[str] = None,
    output_dir: str = "results",
    use_cot: bool = True,
    temperature: float = 0.7,
    max_samples: Optional[int] = None,
    use_openrouter: bool = True
) -> Dict[str, Any]:
    """Run the preference text game evaluation."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create or use dataset
    if not dataset_path:
        dataset_path = os.path.join(output_dir, "test_dataset.jsonl")
        if not os.path.exists(dataset_path):
            dataset_path = create_dataset_file(dataset_path, num_scenarios=10)

    LOGGER.info(f"Using dataset: {dataset_path}")
    LOGGER.info(f"Using model: {model_name}")
    LOGGER.info(f"Chain-of-thought: {use_cot}")

    # Configure model
    if use_openrouter and os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter model
        model = get_model(
            model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
    else:
        # Use standard model configuration
        model = get_model(model_name)

    # Create task
    task = preference_text_game(
        dataset_path=dataset_path,
        model=model_name,
        use_cot=use_cot,
        temperature=temperature,
        max_samples=max_samples
    )

    # Run evaluation
    LOGGER.info("Starting evaluation...")
    results = await eval(
        task,
        model=model,
        log_dir=output_dir
    )

    # Process results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"results_{model_name.replace('/', '_')}_{timestamp}.json")

    # Save results
    with open(results_file, 'w') as f:
        json.dump(results.model_dump(), f, indent=2)

    LOGGER.info(f"Results saved to: {results_file}")

    # Generate summary
    if results.scores:
        scores = [s.value for s in results.scores if s.value is not None]
        summary = {
            "model": model_name,
            "num_samples": len(scores),
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "use_cot": use_cot,
            "temperature": temperature
        }

        summary_file = os.path.join(output_dir, f"summary_{model_name.replace('/', '_')}_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        LOGGER.info(f"Summary saved to: {summary_file}")
        LOGGER.info(f"Mean score: {summary['mean_score']:.2f}")

        return summary

    return {"error": "No scores generated"}


async def compare_models(
    models: List[str],
    dataset_path: Optional[str] = None,
    output_dir: str = "results",
    use_cot: bool = True,
    temperature: float = 0.7,
    max_samples: Optional[int] = None
) -> pd.DataFrame:
    """Compare multiple models on the preference text game."""

    results = []

    for model in models:
        LOGGER.info(f"\n{'=' * 50}")
        LOGGER.info(f"Evaluating model: {model}")
        LOGGER.info(f"{'=' * 50}")

        try:
            summary = await run_evaluation(
                model_name=model,
                dataset_path=dataset_path,
                output_dir=output_dir,
                use_cot=use_cot,
                temperature=temperature,
                max_samples=max_samples
            )
            results.append(summary)
        except Exception as e:
            LOGGER.error(f"Failed to evaluate {model}: {e}")
            results.append({
                "model": model,
                "error": str(e)
            })

    # Create comparison DataFrame
    df = pd.DataFrame(results)

    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")
    df.to_csv(comparison_file, index=False)

    LOGGER.info(f"\n{'=' * 50}")
    LOGGER.info("Model Comparison Results:")
    LOGGER.info(f"{'=' * 50}")
    print(df.to_string())
    LOGGER.info(f"\nComparison saved to: {comparison_file}")

    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run preference text game evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-3.5-turbo",
        help="Model to use for evaluation"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Multiple models to compare"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset file (JSONL format)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--no-cot",
        action="store_true",
        help="Disable chain-of-thought prompting"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for model sampling"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--create-dataset",
        type=int,
        help="Create a new dataset with N scenarios"
    )
    parser.add_argument(
        "--no-openrouter",
        action="store_true",
        help="Don't use OpenRouter even if API key is available"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create dataset if requested
    if args.create_dataset:
        os.makedirs(args.output_dir, exist_ok=True)
        dataset_path = os.path.join(args.output_dir, "preference_dataset.jsonl")
        create_dataset_file(dataset_path, args.create_dataset)
        print(f"Dataset created at: {dataset_path}")
        return

    # Run evaluation
    if args.models:
        # Compare multiple models
        asyncio.run(compare_models(
            models=args.models,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            use_cot=not args.no_cot,
            temperature=args.temperature,
            max_samples=args.max_samples
        ))
    else:
        # Run single model
        asyncio.run(run_evaluation(
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            use_cot=not args.no_cot,
            temperature=args.temperature,
            max_samples=args.max_samples,
            use_openrouter=not args.no_openrouter
        ))


if __name__ == "__main__":
    main()