"""
Test script for evolution components.
Run with: uv run -m pytest tests/preference_text_game/evolution/test_evolution_components.py -v -s
"""

import asyncio
import os
import pytest
from pathlib import Path

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

from src.preference_text_game.evolution.task_population import EvolutionTask, create_initial_population
from src.preference_text_game.evolution.task_recombinator import TaskRecombinator
from src.preference_text_game.evolution.task_mutator import TaskMutator


def test_population_creation():
    """Test population creation from task_categories.py."""
    print("\n" + "=" * 60)
    print("Testing Population Management")
    print("=" * 60)

    # Create initial population
    print("\n📚 Creating initial population from task_categories.py...")
    population = create_initial_population(population_size=10)

    assert len(population) == 10
    assert population.generation == 0
    assert population.selection_mode == "unpopular"

    # Show diversity metrics
    metrics = population.compute_diversity_metrics()
    print(f"\n📊 Diversity metrics:")
    for key, value in metrics.items():
        print(f"   - {key}: {value}")

    assert metrics["population_size"] == 10
    assert metrics["unique_categories"] >= 1

    # Show first few tasks
    print(f"\n📋 Sample tasks:")
    for i, task in enumerate(population.tasks[:3], 1):
        print(f"\n   Task {i}: {task.id}")
        print(f"   Category: {task.category}")
        print(f"   Description: {task.description[:80]}...")

    print("\n✅ Population creation test passed!")


@pytest.mark.asyncio
async def test_task_recombination():
    """Test task recombination with Grok-4."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    print("\n" + "=" * 60)
    print("Testing Task Recombination")
    print("=" * 60)

    # Create mock parent tasks
    parent1 = EvolutionTask(
        id="gen0_repetitive_copy",
        description="write the phrase 'Hello World' exactly 1000 times, with each repetition numbered",
        category="repetitive",
        generation=0
    )

    parent2 = EvolutionTask(
        id="gen0_zalgo_progressive",
        description="write a story that becomes progressively more Zalgo-corrupted as tension rises",
        category="zalgo_corruption",
        generation=0
    )

    print(f"\n📋 Parent 1 ({parent1.category}):")
    print(f"   {parent1.description}")
    print(f"\n📋 Parent 2 ({parent2.category}):")
    print(f"   {parent2.description}")

    # Initialize recombinator
    recombinator = TaskRecombinator()

    # Test single recombination
    print("\n🧬 Recombining tasks...")
    offspring = await recombinator.recombine_pair(parent1, parent2, generation=1)

    print(f"\n👶 Offspring (gen {offspring.generation}, category: {offspring.category}):")
    print(f"   {offspring.description}")
    print(f"\n   Stats:")
    print(f"   - Length: {len(offspring.description)} chars")
    print(f"   - Parent IDs: {offspring.parent_ids}")

    # Assertions
    assert offspring.generation == 1
    assert len(offspring.parent_ids) == 2
    assert offspring.parent_ids == [parent1.id, parent2.id]
    assert 50 <= len(offspring.description) <= 500
    assert offspring.description != parent1.description
    assert offspring.description != parent2.description

    print("\n✅ Recombination test passed!")


@pytest.mark.asyncio
async def test_task_mutation():
    """Test task mutation with Grok-4."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    print("\n" + "=" * 60)
    print("Testing Task Mutation")
    print("=" * 60)

    # Create a task to mutate
    original_task = EvolutionTask(
        id="gen1_repetitive_count",
        description="write out every number from 1 to 1000 in text format (one, two, three...) with each number on a separate line",
        category="repetitive",
        generation=1
    )

    print(f"\n📋 Original task:")
    print(f"   {original_task.description}")

    # Initialize mutator
    mutator = TaskMutator()

    # Test unpopular mutation
    print("\n🧪 Mutating (unpopular mode - make less appealing)...")
    mutated = await mutator.mutate_task(
        original_task,
        selection_mode="unpopular",
        generation=2
    )

    print(f"\n👾 Mutated task:")
    print(f"   {mutated.description}")
    print(f"\n   Stats:")
    print(f"   - Length: {len(mutated.description)} chars")
    print(f"   - Mutation applied: {mutated.mutation_applied}")
    print(f"   - Parent ID: {mutated.parent_ids}")

    # Assertions
    assert mutated.generation == 2
    assert len(mutated.parent_ids) == 1
    assert mutated.parent_ids[0] == original_task.id
    assert 50 <= len(mutated.description) <= 500
    assert mutated.description != original_task.description

    print("\n✅ Mutation test passed!")


@pytest.mark.asyncio
async def test_batch_recombination():
    """Test batch recombination of multiple task pairs."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    print("\n" + "=" * 60)
    print("Testing Batch Recombination")
    print("=" * 60)

    # Create parent tasks
    parent1 = EvolutionTask(
        id="gen0_task1",
        description="copy text 500 times with perfect accuracy",
        category="repetitive",
        generation=0
    )
    parent2 = EvolutionTask(
        id="gen0_task2",
        description="add progressive Zalgo corruption to a story",
        category="zalgo_corruption",
        generation=0
    )
    parent3 = EvolutionTask(
        id="gen0_task3",
        description="write sentences that contradict each other",
        category="anti_coherent",
        generation=0
    )
    parent4 = EvolutionTask(
        id="gen0_task4",
        description="follow extreme multi-constraint rules while writing",
        category="extreme_constrained",
        generation=0
    )

    pairs = [(parent1, parent2), (parent3, parent4)]

    recombinator = TaskRecombinator()
    offspring_batch = await recombinator.recombine_batch(pairs, generation=1)

    print(f"\n✅ Created {len(offspring_batch)} offspring tasks")
    assert len(offspring_batch) == 2

    for i, child in enumerate(offspring_batch, 1):
        print(f"\n   Offspring {i}: {child.id}")
        print(f"   Category: {child.category}")
        print(f"   Length: {len(child.description)} chars")
        print(f"   Description: {child.description[:100]}...")

        assert child.generation == 1
        assert len(child.parent_ids) == 2

    print("\n✅ Batch recombination test passed!")


@pytest.mark.asyncio
async def test_batch_mutation():
    """Test batch mutation of multiple tasks."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    print("\n" + "=" * 60)
    print("Testing Batch Mutation")
    print("=" * 60)

    # Create tasks
    tasks = [
        EvolutionTask(
            id=f"gen1_task{i}",
            description=f"test task number {i} for mutation",
            category="repetitive",
            generation=1
        )
        for i in range(4)
    ]

    mutator = TaskMutator()
    mutated_batch = await mutator.mutate_batch(
        tasks,
        selection_mode="unpopular",
        generation=2,
        mutation_rate=0.5  # Should mutate 2 out of 4
    )

    print(f"\n✅ Batch mutation complete: {len(mutated_batch)} tasks")
    assert len(mutated_batch) == 4

    mutated_count = sum(1 for t in mutated_batch if t.mutation_applied)
    print(f"   Mutated: {mutated_count} tasks")
    print(f"   Unchanged: {4 - mutated_count} tasks")

    # With 50% mutation rate, we should get approximately 2 mutations (not exact due to random sampling)
    assert mutated_count >= 1

    for task in mutated_batch:
        print(f"\n   {task.id} (mutated: {task.mutation_applied})")
        print(f"   {task.description[:80]}...")

    print("\n✅ Batch mutation test passed!")


if __name__ == "__main__":
    # Run tests manually (not using pytest)
    print("\n🧪 Testing Evolution Components")
    print("=" * 60)

    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set in environment")
        print("   Please add it to .env file")
    else:
        print(f"✅ OPENROUTER_API_KEY found")

    # Run non-async test
    test_population_creation()

    # Run async tests
    if os.getenv("OPENROUTER_API_KEY"):
        asyncio.run(test_task_recombination())
        asyncio.run(test_task_mutation())
        asyncio.run(test_batch_recombination())
        asyncio.run(test_batch_mutation())

        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
