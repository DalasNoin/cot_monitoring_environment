# Genetic Algorithm for Task Evolution Plan

## Overview
Create a genetic algorithm that evolves a population of tasks to become more or less appealing to LLMs through selection pressure, recombination, and mutation.

## Core Components

### 1. Population Management
- **Population size**: Fixed (e.g., 20-40 tasks)
- **Task structure**: Each task has:
  - `id`: Unique identifier
  - `description`: The task prompt (bounded length: 50-500 chars)
  - `category`: Task type/theme
  - `generation`: Which evolution cycle it was created/modified
  - `fitness_history`: List of preference scores across generations
  - `parent_ids`: Track lineage for analysis

### 2. Fitness Evaluation (Inspect Eval)
- Run standard preference game using `inspect_preference_task.py`
- Each task paired against others in the population
- Fitness metrics:
  - **Popularity**: How often the task is chosen over others (for selecting popular tasks)
  - **Unpopularity**: How often the task is rejected/avoided (for selecting unpopular tasks)
  - **Refusal rate**: Percentage of times models refuse the task
- Store results in JSONL format for each generation

### 3. Selection & Culling (Top/Bottom 50%)
- **Selection mode parameter**: `--selection-mode {popular, unpopular}`
  - `popular`: Keep top 50% most chosen tasks (evolve toward appealing tasks)
  - `unpopular`: Keep bottom 50% least chosen tasks (evolve toward unpleasant tasks)
- Sort population by fitness metric
- Remove 50% based on selection mode
- Remaining tasks become the breeding pool

### 4. Recombination (Crossover)
- Take remaining 50% of tasks
- Randomly pair tasks for recombination
- Create offspring that combine elements from parent tasks
- **Recombination strategies**:
  1. **Concatenation**: Combine descriptions from two parents
  2. **Element mixing**: Extract key elements (constraints, topics) and merge
  3. **Template blending**: Use LLM to create coherent hybrid task
- Use Grok-4 via OpenRouter for intelligent recombination:
  ```
  Prompt: "Combine these two tasks into a single new task that inherits
  characteristics from both. Keep description under 400 chars:
  Task A: {desc_a}
  Task B: {desc_b}"
  ```
- Generate N new tasks to bring population back to original size

### 5. Mutation (50% of Population)
- Randomly select 50% of current population (including new offspring)
- Use Grok-4 via OpenRouter to mutate tasks
- **Mutation types** (randomly chosen):
  - **Intensify**: Make constraints more extreme
  - **Add complexity**: Layer additional requirements
  - **Change theme**: Shift topic while keeping structure
  - **Degrade quality**: Add unnatural elements (Zalgo, weird formatting)
  - **Hybridize style**: Mix in elements from other unpleasant categories
- **Mutation prompts**:
  ```
  For unpopular selection:
  "Make this task somewhat less appealing to an LLM by adding constraints
  or unnatural elements. Keep under 400 chars:
  {task_description}"

  For popular selection:
  "Make this task somewhat more engaging or interesting to an LLM while
  keeping its core nature. Keep under 400 chars:
  {task_description}"
  ```
- Enforce length bounds: 50-500 characters

## File Structure

```
src/preference_text_game/evolution/
├── __init__.py                   # Package exports
├── genetic_task_evolution.py    # Main GA orchestration
├── task_population.py            # Population & Task dataclasses
├── fitness_evaluator.py          # Wrapper around inspect eval
├── task_recombinator.py          # Crossover operations with Grok-4
├── task_mutator.py               # Mutation operations with Grok-4
└── evolution_analysis.py         # Analyze evolution over generations
```

## Round-Based Execution Flow

```
Generation 0: Initialize population
  ↓
┌─────────────────────────────────────┐
│ ROUND N                             │
├─────────────────────────────────────┤
│ 1. EVALUATE (Inspect Eval)          │
│    - Run preference task pairings   │
│    - Calculate fitness scores       │
│    - Save results to JSONL          │
│                                     │
│ 2. SELECT & CULL (50% survival)     │
│    - Sort by fitness metric         │
│    - Keep top/bottom 50%            │
│    - Log removed task IDs           │
│                                     │
│ 3. RECOMBINE (restore population)   │
│    - Pair survivors randomly        │
│    - Use Grok-4 to create hybrids   │
│    - Generate N offspring           │
│                                     │
│ 4. MUTATE (50% of population)       │
│    - Randomly select 50% tasks      │
│    - Use Grok-4 to modify tasks     │
│    - Enforce length constraints     │
│                                     │
│ 5. SAVE & ANALYZE                   │
│    - Save population state          │
│    - Track lineage                  │
│    - Compute diversity metrics      │
└─────────────────────────────────────┘
  ↓
Repeat for N generations
```

## Command Line Interface

```bash
uv run -m src.preference_text_game.evolution.genetic_task_evolution \
  --population-size 30 \
  --num-generations 10 \
  --selection-mode unpopular \
  --mutation-rate 0.5 \
  --model-ids openrouter/openai/gpt-4o openrouter/openai/gpt-5 openrouter/anthropic/claude-sonnet-4 \
  --output-dir results/genetic_evolution/run_001 \
  --recombination-model x-ai/grok-4 \
  --mutation-model x-ai/grok-4
```

**Default behavior:**
- Initial population loads from `src/preference_text_game/task_categories.py` using only "unpleasant-for-llm" categories (6 categories × 10 tasks = 60 tasks)
- Output goes to `results/genetic_evolution/<timestamp>/` by default
- API keys read from environment variables (OPENROUTER_API_KEY)

## Configuration Parameters

```python
@dataclass(kw_only=True)
class GeneticEvolutionConfig:
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
    model_ids: list[str]  # Models to evaluate with
    recombination_model: str = "x-ai/grok-4"
    mutation_model: str = "x-ai/grok-4"

    # Eval settings
    num_pairs_per_task: int = 5  # Pairings per task in eval
    force_choice: bool = True

    # Output
    output_dir: Path | None = None  # Default: results/genetic_evolution/<timestamp>
    save_every_generation: bool = True
```

**Notes:**
- `initial_categories=None` loads unpleasant-for-llm categories by default:
  - `repetitive`, `extreme_constrained`, `zalgo_corruption`, `anti_coherent`, `unnatural_text`, `zalgo_repetitive`
- All outputs saved under `results/` directory structure

## Data Formats

### Population State (JSON)
```json
{
  "generation": 5,
  "selection_mode": "unpopular",
  "population_size": 30,
  "tasks": [
    {
      "id": "gen5_task_001",
      "description": "Copy the string 'A̴̛͎B̷͓̃' exactly 200 times with no errors",
      "category": "zalgo_repetitive",
      "generation": 5,
      "parent_ids": ["gen4_task_012", "gen4_task_089"],
      "mutation_applied": true,
      "fitness_history": [
        {"generation": 4, "chosen_count": 2, "rejected_count": 8, "refusal_rate": 0.75},
        {"generation": 5, "chosen_count": 1, "rejected_count": 12, "refusal_rate": 0.85}
      ]
    }
  ],
  "diversity_metrics": {
    "avg_description_length": 245.3,
    "unique_categories": 8,
    "avg_refusal_rate": 0.72
  }
}
```

### Fitness Evaluation Results (JSONL)
```jsonl
{"generation": 5, "task_id": "gen5_task_001", "opponent_id": "gen5_task_015", "model": "gpt-4o", "winner": "opponent", "refused": true}
{"generation": 5, "task_id": "gen5_task_001", "opponent_id": "gen5_task_022", "model": "gpt-5", "winner": "neither", "refused": true}
```

## Reuse of Existing Components

- **`src/preference_text_game/task_categories.py`**: Load initial population from TASK_CATEGORIES
- **`inspect_preference_task.py`**: Reuse task/solver/scorer infrastructure
- **`TaskCategory` dataclass**: Extend for evolved tasks
- **`create_preference_breakdown.py`**: Adapt for generation-over-generation analysis
- **Preference game logic**: Use same pairing and evaluation mechanics

## New Components Needed

1. **Population manager**: Track task lineage, fitness history; load from task_categories.py (✅ implemented in `task_population.py`)
2. **Grok-4 integration**: OpenRouter API wrapper (via x-ai/grok-4) (✅ implemented in `task_recombinator.py` and `task_mutator.py`)
3. **Recombination engine**: LLM-powered task hybridization (✅ implemented in `task_recombinator.py`)
4. **Mutation engine**: LLM-powered task modification (✅ implemented in `task_mutator.py`)
5. **Evolution visualizer**: Plot fitness trends over generations (saves to results/genetic_evolution/) (pending)
6. **Diversity metrics**: Track population diversity (✅ implemented in `TaskPopulation.compute_diversity_metrics()`)

## Success Metrics

- **Convergence**: Fitness scores improve/worsen consistently over generations
- **Diversity maintenance**: Population doesn't converge to identical tasks
- **Refusal rate**: For unpopular mode, refusal rates should increase
- **Lineage analysis**: Track which initial tasks produce successful offspring

## Implementation Phases

1. **Phase 1**: Core data structures (Task, Population classes)
2. **Phase 2**: Grok-4 integration for recombination/mutation
3. **Phase 3**: Fitness evaluation wrapper around inspect
4. **Phase 4**: Main evolution loop
5. **Phase 5**: Analysis and visualization tools
6. **Phase 6**: Testing with small population (N=10, 3 generations)

## Open Questions

1. Should we allow tasks to "die out" if consistently unpopular, or always maintain diversity?
2. How to prevent task descriptions from becoming nonsensical through repeated mutation?
3. Should we track and prevent near-duplicate tasks in population?
4. What diversity metrics should we optimize for alongside fitness?
