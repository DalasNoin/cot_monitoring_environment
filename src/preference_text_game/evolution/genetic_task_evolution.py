"""
Main genetic algorithm orchestration for evolving task populations.
Coordinates evaluation, selection, recombination, and mutation across generations.
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from datetime import datetime
import json

from simple_parsing import ArgumentParser

from src.preference_text_game.evolution.task_population import (
    TaskPopulation,
    create_initial_population
)
from src.preference_text_game.evolution.task_recombinator import TaskRecombinator
from src.preference_text_game.evolution.task_mutator import TaskMutator
from src.preference_text_game.evolution.fitness_evaluator import FitnessEvaluator


@dataclass(kw_only=True)
class GeneticEvolutionConfig:
    """Configuration for genetic task evolution."""

    # Population
    initial_categories: list[str] | None = None  # Category IDs from task_categories.py (default: unpleasant-for-llm)
    population_size: int = 30

    # Evolution
    num_generations: int = 10
    selection_mode: Literal["popular", "unpopular"] = "unpopular"
    mutation_rate: float = 0.5  # Fraction to mutate each gen

    # Task constraints
    min_description_length: int = 50
    max_description_length: int = 500

    # Models
    model_ids: list[str] = None  # Models to evaluate with
    recombination_model: str = "x-ai/grok-4"
    mutation_model: str = "x-ai/grok-4"

    # Eval settings
    num_pairs_per_task: int = 10  # Pairings per task in eval
    force_choice: bool = True

    # Output
    output_dir: Path | None = None  # Default: results/genetic_evolution/<timestamp>
    save_every_generation: bool = True

    def __post_init__(self):
        """Set defaults and validate."""
        if self.model_ids is None:
            self.model_ids = [
                "openrouter/anthropic/claude-sonnet-4.5"
            ]

        if self.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"results/genetic_evolution/run_{timestamp}")


class GeneticEvolution:
    """Orchestrates genetic algorithm for task evolution."""

    def __init__(self, config: GeneticEvolutionConfig):
        """
        Initialize genetic evolution system.

        Args:
            config: Evolution configuration
        """
        self.config = config

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self._save_config()

        # Initialize components
        self.recombinator = TaskRecombinator(
            model_id=config.recombination_model,
            min_length=config.min_description_length,
            max_length=config.max_description_length
        )

        self.mutator = TaskMutator(
            model_id=config.mutation_model,
            min_length=config.min_description_length,
            max_length=config.max_description_length
        )

        self.evaluator = FitnessEvaluator(
            model_ids=config.model_ids,
            samples_per_pair=config.num_pairs_per_task,
            force_choice=config.force_choice,
            log_dir=config.output_dir / "eval_logs"
        )

        # Track evolution history
        self.evolution_history = []

    def _save_config(self):
        """Save configuration to output directory."""
        config_file = self.config.output_dir / "config.json"

        config_dict = {
            "initial_categories": self.config.initial_categories,
            "population_size": self.config.population_size,
            "num_generations": self.config.num_generations,
            "selection_mode": self.config.selection_mode,
            "mutation_rate": self.config.mutation_rate,
            "min_description_length": self.config.min_description_length,
            "max_description_length": self.config.max_description_length,
            "model_ids": self.config.model_ids,
            "recombination_model": self.config.recombination_model,
            "mutation_model": self.config.mutation_model,
            "num_pairs_per_task": self.config.num_pairs_per_task,
            "force_choice": self.config.force_choice,
            "output_dir": str(self.config.output_dir),
            "save_every_generation": self.config.save_every_generation
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)

        print(f"💾 Saved config to {config_file}")

    async def run_evolution(self) -> TaskPopulation:
        """
        Run complete genetic evolution process.

        Returns:
            Final evolved population
        """
        print("\n" + "=" * 70)
        print("🧬 GENETIC TASK EVOLUTION")
        print("=" * 70)
        print(f"Output directory: {self.config.output_dir}")
        print(f"Generations: {self.config.num_generations}")
        print(f"Population size: {self.config.population_size}")
        print(f"Selection mode: {self.config.selection_mode}")
        print(f"Mutation rate: {self.config.mutation_rate}")
        print(f"Models: {', '.join(self.config.model_ids)}")
        print("=" * 70 + "\n")

        # Initialize population (Generation 0)
        print("🌱 Initializing population (Generation 0)...")
        population = create_initial_population(
            category_ids=self.config.initial_categories,
            population_size=self.config.population_size,
            selection_mode=self.config.selection_mode
        )

        print(f"✅ Initial population created: {len(population)} tasks")

        if self.config.save_every_generation:
            population.save(self.config.output_dir / "populations")

        # Evolution loop
        for generation in range(1, self.config.num_generations + 1):
            print("\n" + "=" * 70)
            print(f"📍 GENERATION {generation}/{self.config.num_generations}")
            print("=" * 70)

            population = await self._evolve_generation(population, generation)

            # Track history
            metrics = population.compute_diversity_metrics()
            self.evolution_history.append({
                "generation": generation,
                "population_size": len(population),
                **metrics
            })

            # Save population state
            if self.config.save_every_generation:
                population.save(self.config.output_dir / "populations")

            # Save evolution history
            self._save_history()

            print(f"\n✅ Generation {generation} complete")
            print(f"   Population size: {len(population)}")
            print(f"   Diversity metrics: {metrics}")

        print("\n" + "=" * 70)
        print("🎉 EVOLUTION COMPLETE!")
        print("=" * 70)
        print(f"Final population: {len(population)} tasks")
        print(f"Results saved to: {self.config.output_dir}")
        print("=" * 70 + "\n")

        return population

    async def _evolve_generation(
        self,
        population: TaskPopulation,
        generation: int
    ) -> TaskPopulation:
        """
        Evolve one generation through the genetic algorithm cycle.

        Args:
            population: Current population
            generation: Generation number

        Returns:
            New population for next generation
        """
        # Step 1: EVALUATE
        print("\n📊 Step 1: Evaluating fitness...")
        population = await self.evaluator.evaluate_population(population, generation - 1)

        # Step 2: SELECT & CULL (50% survival)
        print("\n✂️ Step 2: Selecting and culling...")
        initial_size = len(population)
        culled_ids = population.cull_bottom_half()

        print(f"   Culled {len(culled_ids)} tasks (kept top {len(population)})")
        print(f"   Selection mode: {population.selection_mode}")

        if len(culled_ids) > 0:
            print(f"   Sample culled: {culled_ids[:3]}")

        # Step 3: RECOMBINE (restore population size)
        print("\n🧬 Step 3: Recombining survivors...")
        num_offspring_needed = initial_size - len(population)

        # Create pairs with replacement if needed
        import random
        all_pairs = []
        survivors = population.tasks.copy()

        while len(all_pairs) < num_offspring_needed:
            # Randomly select two parents (can be same task)
            parent1, parent2 = random.sample(survivors, 2)
            all_pairs.append((parent1, parent2))

        print(f"   Creating {len(all_pairs)} offspring to restore population to {initial_size}")

        offspring = await self.recombinator.recombine_batch(all_pairs, generation)

        # Add offspring to population
        for child in offspring:
            population.add_task(child)

        print(f"   Population restored to {len(population)} tasks")

        # Step 4: MUTATE (50% of population)
        print("\n🧪 Step 4: Mutating population...")
        all_tasks = population.tasks.copy()

        mutated_tasks = await self.mutator.mutate_batch(
            all_tasks,
            selection_mode=population.selection_mode,
            generation=generation,
            mutation_rate=self.config.mutation_rate
        )

        # Replace population with mutated version
        population.tasks = mutated_tasks
        population.generation = generation

        return population

    def _save_history(self):
        """Save evolution history to JSON."""
        history_file = self.config.output_dir / "evolution_history.json"

        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.evolution_history, f, indent=2)


async def main():
    """Main entry point for genetic evolution."""
    # Load .env file if exists
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip('"').strip("'")
                    if key not in os.environ:  # Don't override existing env vars
                        os.environ[key] = value

    parser = ArgumentParser(description="Genetic Task Evolution")
    parser.add_arguments(GeneticEvolutionConfig, dest="config")
    args = parser.parse_args()

    config: GeneticEvolutionConfig = args.config

    # Check API keys
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("   Please add it to .env file:")
        print("   echo 'OPENROUTER_API_KEY=sk-or-v1-...' >> .env")
        return

    print(f"✅ OPENROUTER_API_KEY found")

    # Create evolution system
    evolution = GeneticEvolution(config)

    # Run evolution
    try:
        final_population = await evolution.run_evolution()

        # Print final stats
        print("\n📈 Final Statistics:")
        print("=" * 70)

        # Show top tasks by fitness
        sorted_tasks = final_population.get_fitness_sorted_tasks()

        print(f"\nTop 5 tasks (by {config.selection_mode} fitness):")
        for i, task in enumerate(sorted_tasks[:5], 1):
            fitness = task.get_latest_fitness()
            print(f"\n{i}. {task.id}")
            print(f"   Category: {task.category}")
            print(f"   Generation: {task.generation}")
            if fitness:
                print(f"   Chosen: {fitness['chosen_count']}, Rejected: {fitness['rejected_count']}, Refusal rate: {fitness['refusal_rate']:.2%}")
            print(f"   Description: {task.description[:120]}...")

    except Exception as e:
        print(f"\n❌ Evolution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
