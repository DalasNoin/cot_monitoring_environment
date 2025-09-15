# General Project Guidelines
## Key principles
- All core code resides in the src/ directory.
- All data resides in data/.
- All tests reside in the tests/ directory.
- For any complex functionality in src/, implement tests in tests/. Check existing tests to see if it fits in some existing test file, otherwise create a new one. Tests should not be expensive.
- We run everything with `uv run -m ...`. YOU MUST NEVER USE `python -m ...`.
  - If anything is unclear, look up uv documentation at [https://docs.astral.sh/uv/llms.txt](https://docs.astral.sh/uv/llms.txt).
- Prefer iteration and modularization over code duplication.
- Use descriptive variable names with auxiliary verbs (e.g., is_active, has_permission).
- Use lowercase with underscores for directories and files (e.g., routers/user_routes.py).

## Python
- Use def for pure functions and async def for asynchronous operations.
- All functions that have LLM API requests downstream of them should be async.
- Use type hints for all function signatures. We use dict, |, list, tuple, | None instead of typing.Dict, typing.Union, typing.List, typing.Tuple, typing.Optional.
- Prefer Pydantic models over raw dictionaries or dataclasses for input validation, unless the code is already using certain dictionaries or dataclasses.
  - The `dict` method is deprecated; use `model_dump` instead. Deprecated in Pydantic V2.0 to be removed in V3.0.
- Import at the top of the file.
- Avoid unnecessary curly braces in conditional statements.
- For single-line statements in conditionals, omit curly braces.
- Use concise, one-line syntax for simple conditional statements (e.g., if condition: do_something()).
- **IMPORTANT**: We try to avoid using try-except blocks as much as possible, especially when creating data. LET IT FAIL. Silent failure is the worst, and you can assume basically all failures will be silent unless something fails or you yell about it in multiple places. *Do or do not. There is no try.*

## Ruff
- Your code will be Ruff-formatted by a pre-commit hook.
- We don't care about: E501 (line too long), E402 (module level import not at top of file), E741 (ambiguous variable name), F841 (local variable name is assigned to but never used), F403 (import star), F401 (unused import).
- Use appropriate API clients to make requests to LLMs.
- For LLM requests, use wrapper functions instead of calling API clients directly when available.
- LLM query results should be cached when possible.
- By default, use `tqdm.gather` without batches around async coroutines that involve making LLM requests.
- Take care and log failed requests, or print them prominently. Do not let failed requests slide silently. Also do not let them crash the script.
- Consider internal async parallelization and max-concurrency; thread flags should only control upstream caller concurrency.

### Error Handling
- All LLM API calls should be async with proper error handling
- Failed requests should be logged prominently, not silently ignored
- Processing should continue on individual failures when possible
- Use rate limiting to respect API constraints

## Scripts For Experiments
- Use appropriate base classes for configuring experiments when available.
- Every configuration class should be decorated with `@dataclass(kw_only=True)`. This is to prevent "non-default argument follows default argument" errors.
- We always use ArgumentParser from `simple_parsing` to parse arguments.
- We do everything in JSONL files.
- Some additional common argument patterns:
 - `--num_tasks` is the number of tasks (the size of the prefix of the JSONL file) to use for the experiment.
 - `--num_[sth]` is the number of [sth] to use for the experiment, especially when using subsets of tasks. For example, `--num_pairs 100` with `--num_tasks 50` should pick 100 pairs of tasks out of the first 50 tasks in the JSONL file
 - `--n_repeats` is the number of times to repeat each LLM call (with a different random seed).
- If the script is saving results to a file, you MUST *print the filename of the output file to stdout*, along with a very short description of what is being saved.
- When a script produces data that can be visualized and there is a separate plotting/visualization script, the data production script should print out the command to plot at the end.
- In output filenames, normalize names of models and datasets to not use `/` or whitespace; use underscores instead.
- Before adding new fields to a config class, check if the desired functionality already exists in an ancestor class. It is more likely to exist for general parameters like `model_id`, `temperature`, `num_tasks`, etc.

## Documentation Requirements
- **CRITICAL**: Every script MUST have example commands in its README that include an example input path or clear description of the input data.
- Example commands must show users exactly what data the script operates on and where it comes from.
- Never use vague descriptions like "loads data" - always specify the exact file/directory structure expected.

## Input Data Policy
- **NEVER use default paths for input data in any script**.
- All input data paths must be explicitly provided via command-line arguments.
- Users must always specify `--dataset_path` or equivalent parameters.
- This ensures clarity about what data is being processed and prevents accidental runs on wrong datasets.

## Logging
- Every script should have `LOGGER = logging.getLogger(__name__)` after imports. Do not specify any other logger options in the script.
- Log useful information using `LOGGER.info`. Log suspicious events using `LOGGER.warning`. Log errors using `LOGGER.error`.
- Do not print too many things to stdout unless in debugging mode. Use the library-specific logging functions to log things that are happening in the codebase.

## Dumping data
- The default data dump format for anything over a dataset is JSONL. For example, if you want to dump a dict indexed by an id, use a jsonl where each line is a JSON object, and the first field is `id`.
- The default config dump format is JSON. Dumps should be human-readable, with proper indentation.
- Always round floats to at most 4 decimal places before dumping them in a JSONL file.

## Plots
- Titles and labels are sentence-case.
- Include the model and the relevant config parameters in the plot title. Feel free to add linebreaks in the title if needed.

## Debugging
- We do not use the debugger. We mostly use print statements and iterating.
- When it's not clear why some code is not working, always err on the side of printing very informative stdout so that it is easier to find bugs/issues and debug the code.
- If you find a particular point in code very informative, add `import code; code.interact(local=dict(globals(), **locals()))` there and run the code, or ask the user to run the code and tell you what is happening. This will make you jump into an interactive Python shell where you can inspect variables and run code.
- When printing stuff to stdout, use print(f"{varname=}") to print the variable name and value.
- Do not use fstrings if you're just printing a string without any variables.

- **Background processes**: When testing servers with `&`, always `kill` the actual process PID, not just the shell. Use `ps aux | grep process_name` then `kill <PID>`.

## Testing and Quality

### Running Tests
```bash
# Standard tests
uv run -m pytest tests/ -v -s

# For slow tests (batch API, etc.)
SLOW_TESTS=True uv run -m pytest -v -s -n 6
```

### Code Style
- Line length: 120 characters
- Use ruff for linting and formatting
- Type hints required for all functions
- Async functions for any LLM API interactions

### Test Guidelines
- Write tests for new functions. Do not mock things. If you are testing an LLM-based function, use a real datapoint, and call it with an OpenAI model.
- Before importing external libraries in tests, you may need to import something from `src` first to ensure proper path setup. Do this at the top of the test file when needed.
- For tests that rely on API clients, you need to set up the environment variables for the API keys as required by your specific setup.
- We run tests as follows:
```
uv run -m pytest tests/[test_file].py -v -s
```
or
```
uv run -m pytest tests/[test_file].py::[test_name] -v -s
```
- When writing a function, first write the tests and confirm they fail, before progressing further.
- Make sure the tests pass before committing. If a particular test is failing, think deeply about whether your implementation is correct or the test is wrong.
- You should never mock things to make the tests pass.



# Github and git
- You may use the `gh` CLI to interact with Github, for example, to create a new branch and open a PR. The authentication command is `gh auth login --with-token < <(echo $GH_TOKEN)`.
- Use the `gh` CLI for all GitHub interactions (issues, PRs, checks, etc). When given a GitHub URL, use gh commands to fetch the information.
- You can commit. Make sure you add only the files you have changed as part of a coherent change.
- **ALWAYS run `pre-commit run --all-files` before every commit to avoid formatting issues and unstaged changes after commit. Then run `git status` to see what files were modified by the pre-commit hooks and stage those changes.**
- When asked to commit, commit and push.
- When adding changes to commit and push, you should ALWAYS do `git status` first, then add only the files you want to add in this commit.
- **CRITICAL: ONLY create PRIVATE repositories. NEVER create public repos. User will change to public if needed.**

## Pull Request Planning and Documentation
- **CRITICAL**: When working on PRs, create and maintain a `plan.md` file throughout the development process
- This plan should document:
  - Current implementation progress
  - Key decisions and rationale
  - Test results and verification steps
  - Next steps and remaining work
- **At the end of the PR work**, append the content of `plan.md` to the TOP of `research_log.md` with proper formatting
- Follow the format shown in `research_log.md` with clear headers, status indicators (✅/❌), and structured sections
- This creates a persistent record of all research and development work done on each issue

## Git Worktrees
Git worktrees allow working on multiple branches simultaneously without switching between branches:

### Intelligent Worktree Management
Use the provided `claude-worktree.sh` script for streamlined worktree creation:

```bash
# For GitHub issues (auto-generates branch name using Claude)
source claude-worktree.sh 42

# For specific branch names
source claude-worktree.sh feature-branch
```

**Key features:**
- Creates worktrees in parallel directories (e.g., `../project-name-issue-42`)
- Uses Claude to suggest branch names based on GitHub issue titles
- Handles existing local/remote branches intelligently
- Automatically symlinks shared resources (.venv, .cache, .pytest_cache, uv.lock)
- Copies environment files (.env)
- Sets up GitHub authentication if GH_TOKEN is available

### Automated Cleanup
Use `clean-worktrees.sh` to remove worktrees for merged branches:

```bash
./clean-worktrees.sh
```

**Features:**
- Finds branches from merged PRs using GitHub CLI
- Prompts once for "yes to all" removals
- Uses `trash` command for safe file deletion
- Automatically deletes local branches after removing worktrees
- Lists remaining worktrees when complete

### Worktree Best Practices
- **Shared dependencies**: Symlink `.venv` and lock files to avoid duplicate installations
- **Shared caches**: Symlink `.cache`, `.pytest_cache` for performance
- **Environment consistency**: Copy `.env` files to new worktrees
- **Branch naming**: Use descriptive names like `issue-42-fix-auth-bug`
- **Regular cleanup**: Run cleanup script periodically to maintain repository health

### Benefits Over Branch Switching
- No need to stash/unstash work when switching contexts
- Maintain multiple development environments simultaneously
- Avoid rebuilding caches when switching between tasks
- Enable concurrent testing across different feature branches

# Recent command history
- If running on my machine, you will often find my recent command history helpful if you do not know how to run some command.
- You can do e.g. `rg "uv run -m src." ~/.histfile | tail -n 10` to see the ten most recent commands that start with `uv run -m src.`; or search for `./src/scripts/` to see the most recent bash script runs.

# File management and exploration
- ALWAYS use `trash` instead of `rm` unless the user says otherwise.
- Use `rg` to search for files instead of `grep`.
- USE `tree` FREQUENTLY instead of `ls` to remind yourself of the structure of a directory. You only use `ls` in very rare cases.

# Opening webpages
- If the user provides a documentation in text form (e.g. https://docs.astral.sh/uv/llms.txt), open the links, but keep
- You have a Playwright MCP installed. If not, tell the user to install it using `claude mcp add playwright npx '@playwright/mcp@latest`. This can help you open webpages in the browser and see them as the user sees them.


# Local models

## Server Pattern for Heavy Models
For computationally expensive models (e.g., large vision models, embeddings models) that we have to interact with in a customized fashion (not supported by standard model serving solutions), we use a server-client pattern:

**Design principles:**
   - Run heavy models in a dedicated server process using Starlette/FastAPI (example: `src/clip_server.py`). The default port is 8080.
   - Keep only the computationally expensive operations in the server (e.g., embedding generation is in the server, but similarity computation is in the client)
   - Implement separate endpoints for different model operations (e.g., `/embed_text`, `/embed_image`)
   - Use async/await with queuing to handle requests efficiently
   - Server: Loads model once, generates embeddings/features. Logs all errors with context for debugging
   - Client: Handles all lightweight operations (similarity computation, ranking, filtering)
   - *Performance considerations:*: - Use dynamic batching (accumulate requests before processing, wait for a few miliseconds or for a batch to be full); batch size is specified when running the server but has a default value; -


# Visualization of results
- The default visualization is just stdout. Use stdout if there are only a few numbers to display.
- Whenever needed (e.g. for a dataset), create nice-looking HTML visualizations after processing data; the default location is `localhost:8765/{visualization_name}.html`. Always allow the user to sort/filter the results by key parameters in the displayed HTML.


# Research principles

**Experiment tracking:** After developing code for an experiment, write a one-sentence explanation of the experiment in the relevant `README.md`, together with the full bash command used to run the experiment. Add what the script expects and what it outputs too.

**Visualize early:** Create simple visualizations of results for every experiment. If you are running something, something needs to be visualized. Think about what the most informative plot/table is and create that.

**Trying out with small models on a small dataset:** Run with the default small models and a very small `num_tasks` (e.g. `--num_tasks 2`) before running a full experiment.

**Tell the user the command to run a full experiment:** If asked to run a full experiment, tell the user the command to run a full experiment, and they will run it in another terminal. Use multiline bash and backslashes to make the command copy-pasteable.

**Systematic exploration:** If asked to analyze results, suggest targeted experiments that would provide the next most valuable bits of information.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less. You may suggest the user to change their intention, but do not change their intention without confirming with the user.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
