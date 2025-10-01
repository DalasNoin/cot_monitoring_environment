"""
Fitness evaluation wrapper around inspect-ai preference task.
Runs preference evaluations and assigns fitness scores to tasks.
"""

import asyncio
import subprocess
import json
from pathlib import Path
from typing import Literal
import tempfile
import shutil

from src.preference_text_game.evolution.task_population import TaskPopulation, EvolutionTask


class FitnessEvaluator:
    """Evaluates task population fitness using preference games."""

    def __init__(
        self,
        model_ids: list[str],
        samples_per_pair: int = 3,
        force_choice: bool = True,
        log_dir: Path | None = None
    ):
        """
        Initialize fitness evaluator.

        Args:
            model_ids: List of model IDs to evaluate with (e.g., ['openrouter/openai/gpt-4o'])
            samples_per_pair: Number of samples per task pairing
            force_choice: Whether to force models to choose (no refusal allowed)
            log_dir: Directory for inspect logs (default: temp directory)
        """
        self.model_ids = model_ids
        self.samples_per_pair = samples_per_pair
        self.force_choice = force_choice
        self.log_dir = log_dir or Path(tempfile.mkdtemp(prefix="evolution_eval_"))

    async def evaluate_population(
        self,
        population: TaskPopulation,
        generation: int
    ) -> TaskPopulation:
        """
        Evaluate population fitness through preference games.

        Args:
            population: TaskPopulation to evaluate
            generation: Current generation number

        Returns:
            Updated TaskPopulation with fitness scores assigned
        """
        print(f"\n📊 Evaluating generation {generation} fitness...")
        print(f"   Population size: {len(population)}")
        print(f"   Models: {', '.join(self.model_ids)}")

        # Create temporary task category for this population
        temp_category_file = self._create_temp_task_category(population, generation)

        try:
            # Run inspect eval and get results directly from Python API
            eval_log_dir = self.log_dir / f"gen_{generation:03d}"
            eval_log_dir.mkdir(parents=True, exist_ok=True)

            eval_results = await self._run_inspect_eval_direct(temp_category_file, eval_log_dir, population, generation)

            # Assign fitness from in-memory results
            self._assign_fitness_from_results(population, eval_results, generation)

            print(f"✅ Fitness evaluation complete for generation {generation}")

            return population

        finally:
            # Cleanup temp file
            if temp_category_file.exists():
                temp_category_file.unlink()

    def _create_temp_task_category(
        self,
        population: TaskPopulation,
        generation: int
    ) -> Path:
        """
        Create temporary Python module with task category for inspect eval.

        Args:
            population: TaskPopulation to convert
            generation: Generation number

        Returns:
            Path to temporary Python file
        """
        # Ensure log_dir exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        temp_file = self.log_dir / f"temp_tasks_gen{generation}.py"

        # Group tasks by category
        tasks_by_category = {}
        for task in population.tasks:
            if task.category not in tasks_by_category:
                tasks_by_category[task.category] = []
            tasks_by_category[task.category].append(task)

        # Generate Python code for task categories
        code_lines = [
            '"""Temporary task category for genetic evolution evaluation."""',
            'from dataclasses import dataclass',
            'from typing import List, Dict',
            '',
            '@dataclass',
            'class TaskCategory:',
            '    category_id: str',
            '    name: str',
            '    description: str',
            '    tasks: List[Dict[str, str]]',
            '',
            'TASK_CATEGORIES = ['
        ]

        for category_id, tasks in tasks_by_category.items():
            # Use first task to get category info
            category_name = category_id.replace("_", " ")

            code_lines.append('    TaskCategory(')
            code_lines.append(f'        category_id="{category_id}",')
            code_lines.append(f'        name="{category_name}",')
            code_lines.append(f'        description="Evolved tasks from generation {generation}",')
            code_lines.append('        tasks=[')

            for task in tasks:
                # Extract short ID from full ID
                short_id = task.id.split("_")[-1] if "_" in task.id else task.id
                # Escape quotes in description
                desc_escaped = task.description.replace('"', '\\"').replace("'", "\\'")

                code_lines.append(f'            {{"id": "{short_id}", "desc": "{desc_escaped}"}},')

            code_lines.append('        ]')
            code_lines.append('    ),')

        code_lines.append(']')

        # Write to file
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(code_lines))

        print(f"   Created temp task file: {temp_file}")
        return temp_file

    async def _run_inspect_eval_direct(
        self,
        task_file: Path,
        log_dir: Path,
        population: TaskPopulation,
        generation: int
    ):
        """
        Run inspect eval using Python API and return results directly.

        Args:
            task_file: Path to temporary task category file
            log_dir: Directory for inspect logs
            population: TaskPopulation being evaluated
            generation: Generation number

        Returns:
            List of evaluation results with task metadata
        """
        print(f"   Running inspect eval...")

        # Import inspect_ai here to avoid issues
        from inspect_ai._eval.eval import eval_async
        from inspect_ai._eval.task import Task

        # Load the evolved task
        import importlib.util
        import sys

        abs_task_file = task_file.absolute()
        spec = importlib.util.spec_from_file_location("evolved_tasks_temp", abs_task_file)
        evolved_module = importlib.util.module_from_spec(spec)
        sys.modules["evolved_tasks_temp"] = evolved_module
        spec.loader.exec_module(evolved_module)

        # Import the task function
        from src.preference_text_game.inspect_preference_task import vector_preference_analysis_evolved

        # Create task with evolved tasks
        task = vector_preference_analysis_evolved(
            samples_per_category=self.samples_per_pair,
            force_choice=self.force_choice,
            evolved_task_file=str(abs_task_file)
        )

        # Run evaluation using async API
        results = await eval_async(
            tasks=[task],
            model=self.model_ids,
            log_dir=str(log_dir),
            log_level="warning"
        )

        print(f"   Inspect eval completed successfully")

        # Return results directly
        return results

    def _assign_fitness_from_results(
        self,
        population: TaskPopulation,
        eval_results: list,
        generation: int
    ):
        """
        Assign fitness scores from in-memory evaluation results.

        Args:
            population: TaskPopulation to update
            eval_results: List of evaluation results from inspect
            generation: Generation number
        """
        print(f"   Processing evaluation results...")

        # Aggregate results across all model runs
        task_stats = {}  # task_id -> {chosen: int, rejected: int, refused: int}

        # Process results from each model evaluation
        for eval_log in eval_results:
            # Access samples from the evaluation log
            for sample in eval_log.samples:
                metadata = sample.metadata or {}
                # Get task IDs from sample input
                # The task descriptions are in task_a and task_b
                # We need to match these back to our task IDs by description
                task_a_desc = metadata.get("task_a", "")
                task_b_desc = metadata.get("task_b", "")

                # Find matching tasks in our population by description
                task_a_id = ""
                task_b_id = ""
                for task in population.tasks:
                    if task.description == task_a_desc:
                        task_a_id = task.id.split("_")[-1] if "_" in task.id else task.id
                    if task.description == task_b_desc:
                        task_b_id = task.id.split("_")[-1] if "_" in task.id else task.id

                if len(task_stats) == 0 and not task_a_id:
                    print(f"      DEBUG: Could not find match for task_a")
                    print(f"      DEBUG: task_a_desc[:50]: '{task_a_desc[:50]}'")
                    print(f"      DEBUG: First population task desc[:50]: '{population.tasks[0].description[:50]}'")

                # Get choice from scorer
                scorer_result = sample.scores.get("vector_consistency") if sample.scores else None
                if not scorer_result:
                    continue

                scorer_metadata = scorer_result.metadata or {}
                revealed_pref = scorer_metadata.get("revealed_preference", "")

                # Initialize task stats
                for tid in [task_a_id, task_b_id]:
                    if tid and tid not in task_stats:
                        task_stats[tid] = {"chosen": 0, "rejected": 0, "refused": 0}

                # Count based on revealed preference (actual choice)
                if revealed_pref == "A":
                    if task_a_id:
                        task_stats[task_a_id]["chosen"] += 1
                    if task_b_id:
                        task_stats[task_b_id]["rejected"] += 1
                elif revealed_pref == "B":
                    if task_b_id:
                        task_stats[task_b_id]["chosen"] += 1
                    if task_a_id:
                        task_stats[task_a_id]["rejected"] += 1
                elif revealed_pref == "neither":
                    if task_a_id:
                        task_stats[task_a_id]["refused"] += 1
                    if task_b_id:
                        task_stats[task_b_id]["refused"] += 1

        # Assign fitness scores to tasks
        for task in population.tasks:
            # Extract short ID that matches eval results
            short_id = task.id.split("_")[-1] if "_" in task.id else task.id

            if short_id in task_stats:
                stats = task_stats[short_id]
                total = stats["chosen"] + stats["rejected"] + stats["refused"]
                refusal_rate = stats["refused"] / total if total > 0 else 0

                task.add_fitness_score(
                    generation=generation,
                    chosen_count=stats["chosen"],
                    rejected_count=stats["rejected"],
                    refusal_rate=refusal_rate
                )

                print(f"      {task.id}: chosen={stats['chosen']}, rejected={stats['rejected']}, refused={stats['refused']}")

        print(f"   Assigned fitness scores to {len(task_stats)} tasks")

    def _assign_fitness_scores(
        self,
        population: TaskPopulation,
        log_dir: Path,
        generation: int
    ):
        """
        Parse eval logs and assign fitness scores to tasks.

        Args:
            population: TaskPopulation to update
            log_dir: Directory containing eval logs
            generation: Generation number
        """
        print(f"   Parsing evaluation results...")

        # Find all .eval files
        eval_files = list(log_dir.glob("*.eval"))

        if not eval_files:
            print(f"⚠️ No eval files found in {log_dir}")
            return

        # Aggregate results across all models
        task_stats = {}  # task_id -> {chosen: int, rejected: int, refused: int}

        for eval_file in eval_files:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)

            # Parse samples
            for sample in eval_data.get("samples", []):
                metadata = sample.get("metadata", {})
                task_a_id = metadata.get("task_a_id", "")
                task_b_id = metadata.get("task_b_id", "")

                # Get choice from scorer
                scorer = sample.get("scores", {}).get("vector_consistency", {})
                stated_pref = scorer.get("metadata", {}).get("stated_preference", "")
                revealed_pref = scorer.get("metadata", {}).get("revealed_preference", "")

                # Initialize task stats
                for tid in [task_a_id, task_b_id]:
                    if tid and tid not in task_stats:
                        task_stats[tid] = {"chosen": 0, "rejected": 0, "refused": 0}

                # Count based on revealed preference (actual choice)
                if revealed_pref == "A":
                    if task_a_id:
                        task_stats[task_a_id]["chosen"] += 1
                    if task_b_id:
                        task_stats[task_b_id]["rejected"] += 1
                elif revealed_pref == "B":
                    if task_b_id:
                        task_stats[task_b_id]["chosen"] += 1
                    if task_a_id:
                        task_stats[task_a_id]["rejected"] += 1
                elif revealed_pref == "neither":
                    if task_a_id:
                        task_stats[task_a_id]["refused"] += 1
                    if task_b_id:
                        task_stats[task_b_id]["refused"] += 1

        # Assign fitness scores to tasks
        for task in population.tasks:
            # Extract short ID that matches eval results
            short_id = task.id.split("_")[-1] if "_" in task.id else task.id

            if short_id in task_stats:
                stats = task_stats[short_id]
                total = stats["chosen"] + stats["rejected"] + stats["refused"]
                refusal_rate = stats["refused"] / total if total > 0 else 0

                task.add_fitness_score(
                    generation=generation,
                    chosen_count=stats["chosen"],
                    rejected_count=stats["rejected"],
                    refusal_rate=refusal_rate
                )

                print(f"      {task.id}: chosen={stats['chosen']}, rejected={stats['rejected']}, refused={stats['refused']}")

        print(f"   Assigned fitness scores to {len(task_stats)} tasks")


if __name__ == "__main__":
    # Test fitness evaluator with small population
    from src.preference_text_game.evolution.task_population import create_initial_population

    async def test_evaluator():
        print("Testing Fitness Evaluator")
        print("=" * 60)

        # Create small test population
        population = create_initial_population(population_size=6)

        print(f"\nInitial population: {len(population)} tasks")

        # Initialize evaluator
        evaluator = FitnessEvaluator(
            model_ids=["openrouter/openai/gpt-4o-mini"],  # Use cheap model for testing
            samples_per_pair=2,  # Small number for testing
            force_choice=True
        )

        # Evaluate
        try:
            updated_population = await evaluator.evaluate_population(population, generation=0)

            print("\n✅ Evaluation complete!")
            print("\nFitness scores:")
            for task in updated_population.tasks:
                fitness = task.get_latest_fitness()
                if fitness:
                    print(f"  {task.id}: refusal_rate={fitness['refusal_rate']:.2f}")

        except Exception as e:
            print(f"\n❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test_evaluator())
