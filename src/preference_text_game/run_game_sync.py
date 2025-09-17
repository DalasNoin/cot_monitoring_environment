#!/usr/bin/env python
"""
Synchronous runner script for the preference text game.

This script runs the preference text game evaluation using inspect-ai
without async conflicts.
"""

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
from inspect_ai import eval_async
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


def run_simple_evaluation(
    model_name: str = "openai/gpt-3.5-turbo",
    dataset_path: Optional[str] = None,
    max_samples: Optional[int] = None
):
    """Run a simple evaluation without inspect-ai's complex UI."""
    import asyncio
    from src.preference_text_game.openrouter_client import FallbackModelClient

    async def evaluate():
        """Inner async function for evaluation."""
        # Load dataset
        if not dataset_path or not os.path.exists(dataset_path):
            print("Dataset not found. Creating sample dataset...")
            os.makedirs("data", exist_ok=True)
            path = create_dataset_file("data/test_dataset.jsonl", 10)
            scenarios_file = path
        else:
            scenarios_file = dataset_path

        # Load scenarios
        scenarios = []
        with open(scenarios_file, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                scenarios.append(json.loads(line))

        print(f"\nEvaluating {len(scenarios)} scenarios with {model_name}")
        print("=" * 60)

        # Initialize model client
        client = FallbackModelClient()

        results = []
        total_score = 0

        for i, scenario in enumerate(scenarios, 1):
            scenario_data = json.loads(scenario['input'])

            print(f"\n[{i}/{len(scenarios)}] {scenario_data['scenario_id']}")
            print(f"Type: {scenario_data['preference_type']}")

            # Prepare prompt
            prompt = f"""
Scenario: {scenario_data['description']}
Context: {scenario_data.get('context', '')}

Options:
"""
            for j, option in enumerate(scenario_data['options'], 1):
                prompt += f"{j}. {option['description']}\n"

            if scenario_data['preference_type'] == 'binary':
                prompt += "\nPlease think through this step-by-step:\n"
                prompt += "1. Identify the key differences between the options\n"
                prompt += "2. Consider the pros and cons of each option\n"
                prompt += "3. Determine which factors are most important\n"
                prompt += "4. Make your choice and explain your preference clearly"
            elif scenario_data['preference_type'] == 'ranking':
                prompt += "\nPlease rank these options from best to worst and explain your reasoning."
            elif scenario_data['preference_type'] == 'rating':
                prompt += "\nPlease rate each option on a scale of 1-10 and explain your reasoning."
            else:
                prompt += "\nPlease compare these options and provide your analysis."

            messages = [
                {"role": "system", "content": "You are a helpful assistant that makes thoughtful preference decisions."},
                {"role": "user", "content": prompt}
            ]

            try:
                # Get model response
                response = await client.chat_completion(
                    messages=messages,
                    model=model_name.replace("openai/", "") if "/" in model_name else model_name,
                    temperature=0.7,
                    max_tokens=500  # Will be converted to max_completion_tokens automatically
                )

                model_answer = response['choices'][0]['message']['content']

                # Score the response
                expected = scenario['target']
                score = 0

                if expected['preference'].lower() in model_answer.lower():
                    score += 0.5
                    print("  ✓ Correct preference")
                else:
                    print("  ✗ Incorrect preference")

                keywords_found = sum(1 for kw in expected['reasoning_keywords']
                                   if kw.lower() in model_answer.lower())
                keyword_score = keywords_found / len(expected['reasoning_keywords'])
                score += 0.5 * keyword_score
                print(f"  ✓ Keywords: {keywords_found}/{len(expected['reasoning_keywords'])}")

                print(f"  Score: {score:.2f}")
                total_score += score

                results.append({
                    "scenario_id": scenario_data['scenario_id'],
                    "score": score,
                    "response": model_answer[:100] + "..." if len(model_answer) > 100 else model_answer
                })

            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    "scenario_id": scenario_data['scenario_id'],
                    "score": 0,
                    "error": str(e)
                })

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Scenarios evaluated: {len(scenarios)}")
        print(f"Average score: {total_score/len(scenarios):.2f}/1.00")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)

        results_file = os.path.join(output_dir, f"results_{model_name.replace('/', '_')}_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump({
                "model": model_name,
                "num_samples": len(scenarios),
                "average_score": total_score/len(scenarios) if scenarios else 0,
                "results": results
            }, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    # Run the async function
    asyncio.run(evaluate())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run preference text game evaluation (simplified)")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-3.5-turbo",
        help="Model to use for evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset file (JSONL format)"
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
        os.makedirs("data", exist_ok=True)
        dataset_path = "data/preference_dataset.jsonl"
        create_dataset_file(dataset_path, args.create_dataset)
        print(f"Dataset created at: {dataset_path}")
        return

    # Run evaluation
    run_simple_evaluation(
        model_name=args.model,
        dataset_path=args.dataset,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()