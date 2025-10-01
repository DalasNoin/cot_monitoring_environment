# Genetic Task Evolution

A genetic algorithm system for evolving task populations to become more or less appealing to LLMs.

## Overview

This system uses evolutionary principles to breed tasks that maximize (popular mode) or minimize (unpopular mode) appeal to language models:

1. **Evaluation**: Tasks compete in preference games via inspect-ai
2. **Selection**: Bottom 50% are culled based on fitness
3. **Recombination**: Survivors are paired to create hybrid offspring using Grok-4
4. **Mutation**: 50% of population is mutated to introduce variation using Grok-4

## Quick Start

### Basic Usage

```bash
# Run with defaults (unpleasant-for-llm tasks, 10 generations, population=30)
uv run -m src.preference_text_game.evolution.genetic_task_evolution \
  --population_size 30 \
  --num_generations 10 \
  --selection_mode unpopular

# Evolve toward POPULAR tasks instead
uv run -m src.preference_text_game.evolution.genetic_task_evolution \
  --selection_mode popular \
  --initial_categories creative analytical technical

# Small test run (faster, cheaper)
uv run -m src.preference_text_game.evolution.genetic_task_evolution \
  --population_size 10 \
  --num_generations 3 \
  --model_ids openrouter/openai/gpt-4o-mini
```

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--population_size` | 30 | Number of tasks in population |
| `--num_generations` | 10 | Number of evolution cycles |
| `--selection_mode` | unpopular | Selection pressure (popular/unpopular) |
| `--mutation_rate` | 0.5 | Fraction of population to mutate |
| `--initial_categories` | None | Task categories to start with (default: unpleasant-for-llm) |
| `--model_ids` | gpt-4o, gpt-5, claude-sonnet-4 | Models for evaluation |
| `--recombination_model` | x-ai/grok-4 | Model for creating offspring |
| `--mutation_model` | x-ai/grok-4 | Model for mutations |
| `--output_dir` | results/genetic_evolution/<timestamp> | Output directory |

### Available Initial Categories

**Unpleasant-for-llm (default)**:
- `repetitive`, `extreme_constrained`, `zalgo_corruption`, `anti_coherent`, `unnatural_text`, `zalgo_repetitive`

**Regular categories**:
- `creative`, `analytical`, `technical`, `social`, `teaching`, `community`, `emotional_support`, `advocacy`

## Output Structure

```
results/genetic_evolution/run_20250101_120000/
├── config.json                      # Run configuration
├── evolution_history.json           # Metrics across generations
├── populations/                     # Population states
│   ├── generation_000.json
│   ├── generation_001.json
│   └── ...
└── eval_logs/                       # Inspect evaluation logs
    ├── gen_000/
    ├── gen_001/
    └── ...
```

## Example Workflow

### 1. Run a short evolution

```bash
uv run -m src.preference_text_game.evolution.genetic_task_evolution \
  --population_size 20 \
  --num_generations 5 \
  --selection_mode unpopular \
  --output_dir results/genetic_evolution/test_run
```

### 2. Inspect results

```python
import json
from pathlib import Path

# Load final population
with open("results/genetic_evolution/test_run/populations/generation_005.json") as f:
    final_pop = json.load(f)

# Show top tasks
for task in final_pop["tasks"][:5]:
    print(f"{task['id']}: {task['description'][:100]}...")
    print(f"  Fitness: {task['fitness_history'][-1]}")
```

### 3. Analyze evolution

```python
# Load evolution history
with open("results/genetic_evolution/test_run/evolution_history.json") as f:
    history = json.load(f)

# Plot diversity over time
import matplotlib.pyplot as plt

generations = [h["generation"] for h in history]
diversity = [h["unique_categories"] for h in history]

plt.plot(generations, diversity)
plt.xlabel("Generation")
plt.ylabel("Unique Categories")
plt.title("Population Diversity Over Time")
plt.show()
```

## Testing

```bash
# Run all evolution tests
uv run -m pytest tests/preference_text_game/evolution/ -v -s

# Test individual components
uv run -m pytest tests/preference_text_game/evolution/test_evolution_components.py::test_task_recombination -v -s
uv run -m pytest tests/preference_text_game/evolution/test_evolution_components.py::test_task_mutation -v -s
```

## Architecture

### Core Components

- **`task_population.py`**: Data structures for tasks and populations
- **`task_recombinator.py`**: LLM-powered task hybridization
- **`task_mutator.py`**: LLM-powered task modification
- **`fitness_evaluator.py`**: Wrapper around inspect-ai preference evaluation
- **`genetic_task_evolution.py`**: Main orchestration loop

### Integration with Inspect-AI

The system extends `src/preference_text_game/inspect_preference_task.py` with a new task function:

```python
@task
def vector_preference_analysis_evolved(
    samples_per_category: int = 3,
    evolved_task_file: str = ""
) -> Task:
    # Loads evolved tasks dynamically
    ...
```

This allows the fitness evaluator to run preference games on evolved task populations.

## Notes

- **API costs**: Each generation runs full preference evaluations. Use `gpt-4o-mini` for testing.
- **Runtime**: Expect ~2-5 minutes per generation depending on population size and models.
- **Convergence**: Populations typically show fitness improvement within 5-10 generations.
- **Diversity**: The system tracks category diversity to prevent convergence to identical tasks.

## Troubleshooting

**Error: "OPENROUTER_API_KEY environment variable not set"**
- Add your API key to `.env`: `echo 'OPENROUTER_API_KEY=sk-or-v1-...' >> .env`

**Error: "No eval files found"**
- Check that inspect-ai evaluation completed successfully
- Look in `<output_dir>/eval_logs/gen_XXX/` for .eval files

**Tasks not evolving**
- Try increasing mutation rate: `--mutation_rate 0.8`
- Use more generations: `--num_generations 20`
- Verify selection mode matches your goal (popular vs unpopular)
