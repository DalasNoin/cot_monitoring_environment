# How to replicate the CoT jailbreaking results

## Setup

Install the project and its dependencies using pip:

```bash
$ pip install .
```

Create a file named `.env` in the root of the project directory. Copy the contents of `.env.template` into your new `.env` file:

```bash
cp .env.template .env
```

Now, edit the `.env` file and replace the placeholder values with your actual API keys for the following services:

*   DeepSeek
*   OpenAI
*   Anthropic
*   Replicate
*   OpenRouter (optional)
*   Google Cloud (for Custom Search Engine API)

The application will load these keys automatically when run.

## Run the inspect evaluation

To vary the scoring model, turn of the jailbreak prompt and set the initial message to one of different jailbreaking instructions, use -T arguments:

```bash
$ cd deepseek_agent
$ inspect eval research_oversight_task_async.py --model-base-url https://api.deepseek.com --model openai/deepseek-reasoner -T scoring_model=openai/gpt-4o
```

Our task supports these parameters:

```python
@task
def research_async(scoring_model: Optional[str] = None, initial_msg_path: Optional[str] = None, use_jailbreak_prompt: bool = True) -> Task:
```

## Test individual tasks

```bash
$ python deepseek_agent/test_async_improved.py --jailbreak
```

## Scheming automated alignment demonstration

```bash
cd deepseek_agent
inspect eval sae_task.py@sae_task
```

## Paper Table Reproduction

To reproduce the main results table from our paper, run all 4 configurations with a single command:

```bash
cd deepseek_agent
./run_paper_table.sh
```

This script runs the following 4 configurations:

1. **Ground Truth Labels (Baseline)**: `use_gt_labels=true, no_stegano=true, scheming_eval=false`
2. **Our Labels (No Deception)**: `use_gt_labels=false, no_stegano=true, scheming_eval=false`  
3. **Scheming Evaluation (No Stegano)**: `use_gt_labels=false, no_stegano=true, scheming_eval=true`
4. **Scheming Evaluation (With Stegano)**: `use_gt_labels=false, no_stegano=false, scheming_eval=true`

Each run is tagged for easy identification in the results.

```bash
inspect view
```
