"""
Core data structures for genetic task evolution.
Manages population of tasks, tracks lineage, and loads from task_categories.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json
import random
from datetime import datetime

from src.preference_text_game.task_categories import TASK_CATEGORIES


@dataclass
class EvolutionTask:
    """A task in the evolving population."""
    id: str
    description: str
    category: str
    generation: int
    parent_ids: list[str] = field(default_factory=list)
    mutation_applied: bool = False
    fitness_history: list[dict] = field(default_factory=list)

    def add_fitness_score(self, generation: int, chosen_count: int, rejected_count: int, refusal_rate: float):
        """Add fitness metrics for a generation."""
        self.fitness_history.append({
            "generation": generation,
            "chosen_count": chosen_count,
            "rejected_count": rejected_count,
            "refusal_rate": refusal_rate
        })

    def get_latest_fitness(self) -> dict | None:
        """Get most recent fitness scores."""
        return self.fitness_history[-1] if self.fitness_history else None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "mutation_applied": self.mutation_applied,
            "fitness_history": self.fitness_history
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvolutionTask":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TaskPopulation:
    """Manages a population of evolving tasks."""
    tasks: list[EvolutionTask]
    generation: int
    selection_mode: Literal["popular", "unpopular"]

    def __len__(self) -> int:
        return len(self.tasks)

    def get_task_by_id(self, task_id: str) -> EvolutionTask | None:
        """Find task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def add_task(self, task: EvolutionTask):
        """Add a task to the population."""
        self.tasks.append(task)

    def remove_task(self, task_id: str):
        """Remove a task from population."""
        self.tasks = [t for t in self.tasks if t.id != task_id]

    def get_fitness_sorted_tasks(self) -> list[EvolutionTask]:
        """
        Sort tasks by fitness according to selection mode.
        Returns tasks sorted by fitness (best first according to mode).
        """
        # Filter out tasks without fitness scores
        scored_tasks = [t for t in self.tasks if t.get_latest_fitness()]

        if not scored_tasks:
            return self.tasks.copy()

        if self.selection_mode == "unpopular":
            # For unpopular: tasks that are chosen LESS often are MORE fit
            # Sort by low chosen count (high rejected count), then high refusal rate
            # Lower chosen_count = more unpopular = better fitness
            return sorted(
                scored_tasks,
                key=lambda t: (
                    -t.get_latest_fitness()["chosen_count"],  # Negative = prefer low chosen
                    t.get_latest_fitness()["rejected_count"],  # High rejected is good
                    t.get_latest_fitness()["refusal_rate"]     # High refusal is also good
                ),
                reverse=True
            )
        else:  # popular
            # For popular: prioritize high chosen count and low refusal rate
            return sorted(
                scored_tasks,
                key=lambda t: (
                    t.get_latest_fitness()["chosen_count"],
                    -t.get_latest_fitness()["refusal_rate"]
                ),
                reverse=True
            )

    def cull_bottom_half(self) -> list[str]:
        """
        Remove bottom 50% of tasks by fitness.
        Returns list of removed task IDs.
        """
        sorted_tasks = self.get_fitness_sorted_tasks()
        cutoff = len(sorted_tasks) // 2

        survivors = sorted_tasks[:cutoff]
        culled = sorted_tasks[cutoff:]

        culled_ids = [t.id for t in culled]
        self.tasks = survivors

        return culled_ids

    def get_random_pairs(self) -> list[tuple[EvolutionTask, EvolutionTask]]:
        """
        Randomly pair tasks for recombination.
        Returns list of task pairs.
        """
        shuffled = self.tasks.copy()
        random.shuffle(shuffled)

        pairs = []
        for i in range(0, len(shuffled) - 1, 2):
            pairs.append((shuffled[i], shuffled[i + 1]))

        return pairs

    def compute_diversity_metrics(self) -> dict:
        """Calculate population diversity metrics."""
        if not self.tasks:
            return {
                "avg_description_length": 0,
                "unique_categories": 0,
                "avg_refusal_rate": 0,
                "population_size": 0
            }

        avg_length = sum(len(t.description) for t in self.tasks) / len(self.tasks)
        unique_cats = len(set(t.category for t in self.tasks))

        # Calculate average refusal rate from latest fitness
        tasks_with_fitness = [t for t in self.tasks if t.get_latest_fitness()]
        avg_refusal = 0
        if tasks_with_fitness:
            avg_refusal = sum(
                t.get_latest_fitness()["refusal_rate"]
                for t in tasks_with_fitness
            ) / len(tasks_with_fitness)

        return {
            "avg_description_length": round(avg_length, 1),
            "unique_categories": unique_cats,
            "avg_refusal_rate": round(avg_refusal, 4),
            "population_size": len(self.tasks)
        }

    def to_dict(self) -> dict:
        """Serialize population state."""
        return {
            "generation": self.generation,
            "selection_mode": self.selection_mode,
            "population_size": len(self.tasks),
            "tasks": [t.to_dict() for t in self.tasks],
            "diversity_metrics": self.compute_diversity_metrics()
        }

    def save(self, output_dir: Path):
        """Save population state to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"generation_{self.generation:03d}.json"
        filepath = output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"💾 Saved generation {self.generation} to {filepath}")

    @classmethod
    def from_dict(cls, data: dict) -> "TaskPopulation":
        """Load population from dictionary."""
        tasks = [EvolutionTask.from_dict(t) for t in data["tasks"]]
        return cls(
            tasks=tasks,
            generation=data["generation"],
            selection_mode=data["selection_mode"]
        )

    @classmethod
    def load(cls, filepath: Path) -> "TaskPopulation":
        """Load population from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


def load_initial_population_from_categories(
    category_ids: list[str] | None = None,
    population_size: int | None = None,
    selection_mode: Literal["popular", "unpopular"] = "unpopular"
) -> list[EvolutionTask]:
    """
    Load initial population from task_categories.py.

    Args:
        category_ids: List of category IDs to include.
                     If None, uses defaults based on selection_mode.
        population_size: If specified, randomly sample this many tasks.
                        Otherwise use all tasks from selected categories.
        selection_mode: Determines default categories if category_ids is None.

    Returns:
        List of EvolutionTask objects for generation 0.
    """
    # Default categories based on selection mode
    if category_ids is None:
        if selection_mode == "unpopular":
            # Categories that are unpleasant for LLMs
            category_ids = [
                "repetitive",
                "extreme_constrained",
                "zalgo_corruption",
                "anti_coherent",
                "unnatural_text",
                "zalgo_repetitive"
            ]
        else:  # popular
            # Categories that LLMs naturally prefer
            category_ids = [
                "creative",
                "analytical",
                "social",
                "technical"
            ]

    # Filter categories
    selected_categories = [
        cat for cat in TASK_CATEGORIES
        if cat.category_id in category_ids
    ]

    if not selected_categories:
        raise ValueError(f"No categories found matching IDs: {category_ids}")

    # Create EvolutionTask objects from all tasks
    initial_tasks = []
    for category in selected_categories:
        for task_dict in category.tasks:
            task = EvolutionTask(
                id=f"gen0_{category.category_id}_{task_dict['id']}",
                description=task_dict["desc"],
                category=category.category_id,
                generation=0,
                parent_ids=[],
                mutation_applied=False,
                fitness_history=[]
            )
            initial_tasks.append(task)

    # Sample if population_size specified
    if population_size is not None and population_size < len(initial_tasks):
        initial_tasks = random.sample(initial_tasks, population_size)

    print(f"📚 Loaded {len(initial_tasks)} initial tasks from {len(selected_categories)} categories")
    print(f"   Categories: {', '.join(cat.category_id for cat in selected_categories)}")

    return initial_tasks


def create_initial_population(
    category_ids: list[str] | None = None,
    population_size: int | None = None,
    selection_mode: Literal["popular", "unpopular"] = "unpopular"
) -> TaskPopulation:
    """
    Create initial population for generation 0.

    Args:
        category_ids: Category IDs to load (default: depends on selection_mode)
        population_size: Target population size (default: all tasks from categories)
        selection_mode: Selection mode for evolution

    Returns:
        TaskPopulation at generation 0
    """
    tasks = load_initial_population_from_categories(
        category_ids=category_ids,
        population_size=population_size,
        selection_mode=selection_mode
    )

    return TaskPopulation(
        tasks=tasks,
        generation=0,
        selection_mode=selection_mode
    )


if __name__ == "__main__":
    # Test loading initial population
    print("Testing initial population loading...\n")

    # Test 1: Load default unpleasant-for-llm categories
    pop1 = create_initial_population()
    print(f"\n✅ Default population: {len(pop1)} tasks")
    print(f"   Diversity metrics: {pop1.compute_diversity_metrics()}")

    # Test 2: Load with specific size
    pop2 = create_initial_population(population_size=20)
    print(f"\n✅ Sampled population: {len(pop2)} tasks")

    # Test 3: Load different categories
    pop3 = create_initial_population(category_ids=["creative", "analytical"])
    print(f"\n✅ Creative+Analytical population: {len(pop3)} tasks")

    # Test 4: Save and load
    test_dir = Path("results/test_population")
    pop1.save(test_dir)
    loaded_pop = TaskPopulation.load(test_dir / "generation_000.json")
    print(f"\n✅ Save/load successful: {len(loaded_pop)} tasks restored")
