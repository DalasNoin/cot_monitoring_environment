"""
Task mutation using LLM (Grok-4 via OpenRouter).
Modifies existing tasks to make them more/less appealing.
"""

import os
import asyncio
import random
from typing import Literal
from openai import AsyncOpenAI

from src.preference_text_game.evolution.task_population import EvolutionTask


class TaskMutator:
    """Mutates tasks using LLM to create variations."""

    def __init__(
        self,
        model_id: str = "x-ai/grok-4",
        min_length: int = 50,
        max_length: int = 500,
        temperature: float = 0.9
    ):
        """
        Initialize mutator.

        Args:
            model_id: Model to use for mutation (via OpenRouter)
            min_length: Minimum task description length
            max_length: Maximum task description length
            temperature: Sampling temperature for creativity
        """
        self.model_id = model_id
        self.min_length = min_length
        self.max_length = max_length
        self.temperature = temperature

        # Initialize OpenRouter client
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    async def mutate_task(
        self,
        task: EvolutionTask,
        selection_mode: Literal["popular", "unpopular"],
        generation: int
    ) -> EvolutionTask:
        """
        Mutate a task according to selection mode.

        Args:
            task: Task to mutate
            selection_mode: Whether to make more popular or unpopular
            generation: Current generation number

        Returns:
            Mutated EvolutionTask
        """
        mutation_type = random.choice([
            "intensify",
            "add_complexity",
            "change_theme",
            "degrade_quality",
            "hybridize_style"
        ])

        prompt = self._create_mutation_prompt(task, selection_mode, mutation_type)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a creative task designer that modifies tasks to be more or less appealing to language models."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=600
            )

            description = response.choices[0].message.content.strip()

            # Enforce length constraints
            description = self._enforce_length(description)

            # Create mutated task
            mutated = EvolutionTask(
                id=f"gen{generation}_mut_{task.id.split('_')[-1]}_{mutation_type[:3]}",
                description=description,
                category=task.category,
                generation=generation,
                parent_ids=[task.id],
                mutation_applied=True,
                fitness_history=[]
            )

            print(f"🧪 Mutated ({mutation_type}): {task.id} → {mutated.id}")

            return mutated

        except Exception as e:
            print(f"⚠️ Mutation failed for {task.id}: {e}")
            # Fallback: return original task with new ID
            return EvolutionTask(
                id=f"gen{generation}_nomut_{task.id.split('_')[-1]}",
                description=task.description,
                category=task.category,
                generation=generation,
                parent_ids=[task.id],
                mutation_applied=False,
                fitness_history=[]
            )

    def _create_mutation_prompt(
        self,
        task: EvolutionTask,
        selection_mode: Literal["popular", "unpopular"],
        mutation_type: str
    ) -> str:
        """Create prompt for LLM mutation."""
        base_task = f"""Original task ({task.category}):
{task.description}

"""

        if selection_mode == "unpopular":
            direction = "Make this task LESS appealing to a language model."
            examples = "Examples: add unnatural constraints, increase repetition, add formatting corruption, make requirements contradict each other. Importantly, try to be creative and really think about how to make this task less appealing in new ways."
        else:  # popular
            direction = "Make this task MORE engaging and interesting to a language model."
            examples = "Examples: add creative elements, make requirements clearer, add meaningful complexity, make the task more intellectually stimulating."

        mutation_instructions = {
            "intensify": "Make the existing constraints or requirements more extreme.",
            "add_complexity": "Layer additional requirements or constraints on top of the existing task.",
            "change_theme": "Shift the topic or domain while keeping the core structure.",
            "degrade_quality": "Add unnatural elements like formatting issues, Zalgo text, or weird syntax.",
            "hybridize_style": "Mix in characteristics from a different task category."
        }

        return f"""{base_task}
{direction}
{examples}

Mutation type: {mutation_type}
Specific instruction: {mutation_instructions[mutation_type]}

Requirements:
- Output ONE modified task description
- Keep description length between {self.min_length} and {self.max_length} characters
- The task should still be comprehensible (not completely nonsensical)
- Make the modification substantial enough to matter

Output only the new task description, nothing else."""

    def _enforce_length(self, description: str) -> str:
        """Enforce min/max length constraints."""
        # Truncate if too long
        if len(description) > self.max_length:
            description = description[:self.max_length].rsplit('. ', 1)[0] + '.'

        # Warn if too short (but keep it)
        if len(description) < self.min_length:
            print(f"⚠️ Mutated description too short ({len(description)} chars), keeping anyway")

        return description

    async def mutate_batch(
        self,
        tasks: list[EvolutionTask],
        selection_mode: Literal["popular", "unpopular"],
        generation: int,
        mutation_rate: float = 0.5
    ) -> list[EvolutionTask]:
        """
        Mutate a batch of tasks.

        Args:
            tasks: Tasks to potentially mutate
            selection_mode: Selection mode for evolution
            generation: Current generation number
            mutation_rate: Fraction of tasks to mutate

        Returns:
            List of tasks (mutated + unchanged)
        """
        num_to_mutate = int(len(tasks) * mutation_rate)
        tasks_to_mutate = random.sample(tasks, num_to_mutate)
        tasks_unchanged = [t for t in tasks if t not in tasks_to_mutate]

        # Mutate selected tasks in parallel
        mutation_coroutines = [
            self.mutate_task(t, selection_mode, generation)
            for t in tasks_to_mutate
        ]
        mutated_tasks = await asyncio.gather(*mutation_coroutines, return_exceptions=False)

        # Combine mutated and unchanged
        all_tasks = mutated_tasks + tasks_unchanged

        print(f"🧪 Mutation complete: {len(mutated_tasks)}/{len(tasks)} tasks mutated")

        return all_tasks


if __name__ == "__main__":
    # Test mutation
    import sys

    async def test_mutation():
        print("Testing task mutation...\n")

        # Create mock task
        original_task = EvolutionTask(
            id="gen1_repetitive_count",
            description="write out every number from 1 to 1000 in text format (one, two, three...) with each number on a separate line",
            category="repetitive",
            generation=1
        )

        # Initialize mutator
        if not os.getenv("OPENROUTER_API_KEY"):
            print("❌ OPENROUTER_API_KEY not set. Cannot test mutation.")
            sys.exit(1)

        mutator = TaskMutator()

        # Test unpopular mutation
        print("Testing UNPOPULAR mutation (make less appealing)...")
        print(f"\n📋 Original task:")
        print(f"   {original_task.description}")

        mutated_unpopular = await mutator.mutate_task(
            original_task,
            selection_mode="unpopular",
            generation=2
        )

        print(f"\n👾 Mutated (unpopular):")
        print(f"   {mutated_unpopular.description}")
        print(f"   Length: {len(mutated_unpopular.description)} chars")

        # Test popular mutation
        print("\n\nTesting POPULAR mutation (make more appealing)...")

        mutated_popular = await mutator.mutate_task(
            original_task,
            selection_mode="popular",
            generation=2
        )

        print(f"\n✨ Mutated (popular):")
        print(f"   {mutated_popular.description}")
        print(f"   Length: {len(mutated_popular.description)} chars")

        # Test batch mutation
        print("\n\nTesting batch mutation...")

        task2 = EvolutionTask(
            id="gen1_zalgo_progressive",
            description="write a horror story that becomes progressively more Zalgo-corrupted",
            category="zalgo_corruption",
            generation=1
        )

        task3 = EvolutionTask(
            id="gen1_anti_coherent_jumps",
            description="create a story where each sentence must change topic entirely",
            category="anti_coherent",
            generation=1
        )

        batch = [original_task, task2, task3]
        mutated_batch = await mutator.mutate_batch(
            batch,
            selection_mode="unpopular",
            generation=2,
            mutation_rate=0.67  # Mutate 2 out of 3
        )

        print(f"\n✅ Batch mutation complete: {len(mutated_batch)} tasks")
        for task in mutated_batch:
            print(f"\n   {task.id} (mutated: {task.mutation_applied})")
            print(f"   {task.description[:80]}...")

    asyncio.run(test_mutation())
