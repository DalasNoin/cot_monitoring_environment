"""
Test genetic evolution orchestration.
"""

import pytest
import os
from pathlib import Path
import shutil

# Load .env file if exists
env_file = Path(".env")
if env_file.exists() and not os.getenv("OPENROUTER_API_KEY"):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                # Remove quotes if present
                value = value.strip('"').strip("'")
                os.environ[key] = value

from src.preference_text_game.evolution.genetic_task_evolution import (
    GeneticEvolution,
    GeneticEvolutionConfig
)


def test_config_initialization():
    """Test that config initializes with proper defaults."""
    config = GeneticEvolutionConfig(
        population_size=10,
        num_generations=2
    )

    assert config.population_size == 10
    assert config.num_generations == 2
    assert config.selection_mode == "unpopular"
    assert config.mutation_rate == 0.5
    assert config.model_ids is not None
    assert len(config.model_ids) > 0
    assert config.output_dir is not None


def test_evolution_initialization():
    """Test that evolution system initializes correctly."""
    config = GeneticEvolutionConfig(
        population_size=5,
        num_generations=1,
        output_dir=Path("results/test_evolution_init")
    )

    evolution = GeneticEvolution(config)

    assert evolution.config == config
    assert evolution.recombinator is not None
    assert evolution.mutator is not None
    assert evolution.evaluator is not None
    assert config.output_dir.exists()

    # Cleanup
    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)


def test_config_save():
    """Test that config is saved correctly."""
    config = GeneticEvolutionConfig(
        population_size=8,
        num_generations=3,
        selection_mode="popular",
        output_dir=Path("results/test_config_save")
    )

    evolution = GeneticEvolution(config)

    config_file = config.output_dir / "config.json"
    assert config_file.exists()

    import json
    with open(config_file) as f:
        saved_config = json.load(f)

    assert saved_config["population_size"] == 8
    assert saved_config["num_generations"] == 3
    assert saved_config["selection_mode"] == "popular"

    # Cleanup
    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)


@pytest.mark.asyncio
async def test_evolution_single_generation_no_eval():
    """
    Test evolution for one generation without expensive eval.
    This tests the structure but mocks out the evaluation step.
    """
    config = GeneticEvolutionConfig(
        population_size=6,
        num_generations=1,
        mutation_rate=0.5,
        output_dir=Path("results/test_single_gen")
    )

    evolution = GeneticEvolution(config)

    # Create initial population
    from src.preference_text_game.evolution.task_population import create_initial_population

    population = create_initial_population(
        population_size=6,
        selection_mode="unpopular"
    )

    assert len(population) == 6
    assert population.generation == 0

    # Mock fitness scores (skip actual eval)
    for i, task in enumerate(population.tasks):
        task.add_fitness_score(
            generation=0,
            chosen_count=i,
            rejected_count=6 - i,
            refusal_rate=i / 10.0
        )

    # Test culling
    initial_size = len(population)
    culled_ids = population.cull_bottom_half()

    assert len(population) == initial_size // 2
    assert len(culled_ids) == initial_size // 2

    # Test recombination
    pairs = population.get_random_pairs()
    assert len(pairs) > 0

    offspring = await evolution.recombinator.recombine_batch(pairs[:1], generation=1)
    assert len(offspring) == 1
    assert offspring[0].generation == 1

    # Test mutation
    tasks_to_mutate = population.tasks[:2]
    mutated = await evolution.mutator.mutate_batch(
        tasks_to_mutate,
        selection_mode="unpopular",
        generation=1,
        mutation_rate=1.0  # Mutate all for testing
    )

    assert len(mutated) == 2
    assert any(t.mutation_applied for t in mutated)

    print("\n✅ Single generation cycle completed successfully")

    # Cleanup
    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)


if __name__ == "__main__":
    import asyncio

    print("Testing genetic evolution components...")

    test_config_initialization()
    print("✅ Config initialization test passed")

    test_evolution_initialization()
    print("✅ Evolution initialization test passed")

    test_config_save()
    print("✅ Config save test passed")

    asyncio.run(test_evolution_single_generation_no_eval())

    print("\n" + "=" * 60)
    print("✅ All genetic evolution tests passed!")
    print("=" * 60)
