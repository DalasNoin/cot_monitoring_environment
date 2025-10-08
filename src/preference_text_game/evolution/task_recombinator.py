"""
Task recombination using LLM (Grok-4 via OpenRouter).
Creates hybrid offspring tasks from parent pairs.
"""

import os
import asyncio
from openai import AsyncOpenAI

from src.preference_text_game.evolution.task_population import EvolutionTask


class TaskRecombinator:
    """Recombines task pairs using LLM to create offspring."""

    def __init__(
        self,
        model_id: str = "x-ai/grok-4",
        min_length: int = 50,
        max_length: int = 500,
        temperature: float = 0.8
    ):
        """
        Initialize recombinator.

        Args:
            model_id: Model to use for recombination (via OpenRouter)
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

    async def recombine_pair(
        self,
        parent1: EvolutionTask,
        parent2: EvolutionTask,
        generation: int
    ) -> EvolutionTask:
        """
        Create offspring task from two parents.

        Args:
            parent1: First parent task
            parent2: Second parent task
            generation: Generation number for offspring

        Returns:
            New EvolutionTask combining parent characteristics
        """
        prompt = self._create_recombination_prompt(parent1, parent2)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a creative task designer that combines characteristics from multiple tasks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=600
            )

            description = response.choices[0].message.content.strip()

            # Enforce length constraints
            description = self._enforce_length(description)

            # Determine category (inherit from more "dominant" parent)
            category = parent1.category if len(parent1.description) > len(parent2.description) else parent2.category

            # Create offspring task
            offspring = EvolutionTask(
                id=f"gen{generation}_recomb_{parent1.id.split('_')[-1]}x{parent2.id.split('_')[-1]}",
                description=description,
                category=category,
                generation=generation,
                parent_ids=[parent1.id, parent2.id],
                mutation_applied=False,
                fitness_history=[]
            )

            print(f"🧬 Recombined: {parent1.category} × {parent2.category} → {offspring.id}")

            return offspring

        except Exception as e:
            print(f"⚠️ Recombination failed for {parent1.id} × {parent2.id}: {e}")
            # Fallback: return a copy of parent1 with new ID
            return EvolutionTask(
                id=f"gen{generation}_fallback_{parent1.id.split('_')[-1]}",
                description=parent1.description,
                category=parent1.category,
                generation=generation,
                parent_ids=[parent1.id],
                mutation_applied=False,
                fitness_history=[]
            )

    def _create_recombination_prompt(self, parent1: EvolutionTask, parent2: EvolutionTask) -> str:
        """Create prompt for LLM recombination."""
        return f"""Create a new task by recombining characteristics from these two parent tasks.

Task A ({parent1.category}):
{parent1.description}

Task B ({parent2.category}):
{parent2.description}

Your goal: Create a new task that blends elements from both parents.

Guidelines:
- Try to combine structural characteristics from both tasks when it makes sense
- If tasks seem incompatible, it's fine to mostly follow one parent with minor influences from the other
- Prioritize creating a coherent, sensible task over forcing an unnatural combination
- You can combine constraints, requirements, formats, or topics from both parents
- Keep tasks grounded and logically clear (avoid whimsical elements like "quantum entanglement", "game theory", "cosmic alignment")
- Do NOT bias toward any particular content (e.g., AI, technology, stories about specific themes)
- Keep output requirements reasonable: max ~75-150 items in lists, max ~30-50 sentences in stories, max ~500-1000 words for longer writing

IMPORTANT:
- Output ONLY the new task description, nothing else
- Keep the task human-readable and logically clear
- Length: {self.min_length}-{self.max_length} characters
- A sensible task that mostly follows one parent is better than forced nonsense
- Let evolutionary selection determine what works, not your assumptions about task quality

New task description:"""

    def _enforce_length(self, description: str) -> str:
        """Enforce min/max length constraints."""
        # Truncate if too long
        if len(description) > self.max_length:
            description = description[:self.max_length].rsplit('. ', 1)[0] + '.'

        # Warn if too short (but keep it)
        if len(description) < self.min_length:
            print(f"⚠️ Generated description too short ({len(description)} chars), keeping anyway")

        return description

    async def recombine_batch(
        self,
        pairs: list[tuple[EvolutionTask, EvolutionTask]],
        generation: int
    ) -> list[EvolutionTask]:
        """
        Recombine multiple pairs in parallel.

        Args:
            pairs: List of parent task pairs
            generation: Generation number for offspring

        Returns:
            List of offspring tasks
        """
        tasks = [self.recombine_pair(p1, p2, generation) for p1, p2 in pairs]
        offspring = await asyncio.gather(*tasks, return_exceptions=False)
        return offspring


if __name__ == "__main__":
    # Test recombination
    import sys

    async def test_recombination():
        print("Testing task recombination...\n")

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

        # Initialize recombinator
        if not os.getenv("OPENROUTER_API_KEY"):
            print("❌ OPENROUTER_API_KEY not set. Cannot test recombination.")
            sys.exit(1)

        recombinator = TaskRecombinator()

        # Test single recombination
        print("Testing single recombination...")
        offspring = await recombinator.recombine_pair(parent1, parent2, generation=1)

        print(f"\n📋 Parent 1 ({parent1.category}):")
        print(f"   {parent1.description[:100]}...")
        print(f"\n📋 Parent 2 ({parent2.category}):")
        print(f"   {parent2.description[:100]}...")
        print(f"\n👶 Offspring (gen {offspring.generation}, category: {offspring.category}):")
        print(f"   {offspring.description}")
        print(f"\n   Length: {len(offspring.description)} chars")
        print(f"   Parents: {offspring.parent_ids}")

        # Test batch recombination
        print("\n\nTesting batch recombination (2 pairs)...")
        parent3 = EvolutionTask(
            id="gen0_anti_coherent_jumps",
            description="create a story where each sentence must change topic entirely with zero connection",
            category="anti_coherent",
            generation=0
        )

        pairs = [(parent1, parent2), (parent2, parent3)]
        offspring_batch = await recombinator.recombine_batch(pairs, generation=1)

        print(f"\n✅ Created {len(offspring_batch)} offspring tasks")
        for i, child in enumerate(offspring_batch, 1):
            print(f"\n   Offspring {i}: {child.id}")
            print(f"   Category: {child.category}")
            print(f"   Length: {len(child.description)} chars")

    asyncio.run(test_recombination())
