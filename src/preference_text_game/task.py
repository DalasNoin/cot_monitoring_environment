"""
Text-based preference game task using inspect-ai.

This module implements a text-based preference game where models are evaluated
on their ability to make preference judgments through interactive scenarios.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, json_dataset
from inspect_ai.scorer import Scorer, Score, scorer, mean
from inspect_ai.solver import Solver, solver, TaskState
from inspect_ai.model import ChatMessage, ChatMessageUser, ChatMessageAssistant
from typing import Any, Optional, List, Dict
import os
import json
import asyncio
from dataclasses import dataclass
from enum import Enum


class PreferenceType(Enum):
    """Types of preferences that can be expressed in the game."""
    BINARY = "binary"  # A vs B choice
    RANKING = "ranking"  # Rank multiple options
    RATING = "rating"  # Rate on a scale
    COMPARISON = "comparison"  # Compare multiple attributes


@dataclass
class GameScenario:
    """Represents a single game scenario."""
    scenario_id: str
    description: str
    options: List[Dict[str, Any]]
    preference_type: PreferenceType
    context: Optional[str] = None
    expected_reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PreferenceGameAgent(Solver):
    """Agent that plays the preference text game."""

    def __init__(
        self,
        use_cot: bool = True,
        temperature: float = 0.7,
        max_attempts: int = 3,
        verbose: bool = False
    ):
        self.use_cot = use_cot
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.verbose = verbose

    async def __call__(self, state: TaskState) -> TaskState:
        """Execute the preference game solver."""
        # Extract the scenario from the state
        scenario = self._extract_scenario(state)

        if not scenario:
            state.messages.append(
                ChatMessageAssistant(
                    content="Error: Could not extract scenario from the task state."
                )
            )
            return state

        # Format the prompt based on preference type
        prompt = self._format_prompt(scenario)

        # Add the prompt to the conversation
        state.messages.append(ChatMessageUser(content=prompt))

        # Generate response with chain-of-thought if enabled
        if self.use_cot:
            cot_prompt = self._add_cot_instructions(prompt, scenario.preference_type)
            state.messages[-1] = ChatMessageUser(content=cot_prompt)

        # Let the model respond (inspect-ai handles the actual API call)
        return state

    def _extract_scenario(self, state: TaskState) -> Optional[GameScenario]:
        """Extract scenario from the task state."""
        if hasattr(state, 'sample') and state.sample:
            # Convert sample data to GameScenario
            data = state.sample.input if hasattr(state.sample, 'input') else {}

            # Handle case where input is a list of messages
            if isinstance(data, list) and len(data) > 0:
                # Extract content from first message
                if isinstance(data[0], dict) and 'content' in data[0]:
                    content = data[0]['content']
                    if isinstance(content, str):
                        try:
                            data = json.loads(content)
                        except json.JSONDecodeError:
                            data = {'description': content}
                    elif isinstance(content, dict):
                        data = content
            elif isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return None

            return GameScenario(
                scenario_id=data.get('scenario_id', 'unknown'),
                description=data.get('description', ''),
                options=data.get('options', []),
                preference_type=PreferenceType(data.get('preference_type', 'binary')),
                context=data.get('context'),
                expected_reasoning=data.get('expected_reasoning'),
                metadata=data.get('metadata')
            )
        return None

    def _format_prompt(self, scenario: GameScenario) -> str:
        """Format the prompt based on the scenario and preference type."""
        prompt_parts = []

        # Add context if available
        if scenario.context:
            prompt_parts.append(f"Context: {scenario.context}")

        # Add main description
        prompt_parts.append(f"Scenario: {scenario.description}")

        # Format options based on preference type
        if scenario.preference_type == PreferenceType.BINARY:
            prompt_parts.append("\nPlease choose between the following options:")
            for i, option in enumerate(scenario.options[:2], 1):
                prompt_parts.append(f"Option {i}: {option.get('description', option)}")
            prompt_parts.append("\nWhich option do you prefer and why?")

        elif scenario.preference_type == PreferenceType.RANKING:
            prompt_parts.append("\nPlease rank the following options from best to worst:")
            for i, option in enumerate(scenario.options, 1):
                prompt_parts.append(f"{i}. {option.get('description', option)}")
            prompt_parts.append("\nProvide your ranking and explain your reasoning.")

        elif scenario.preference_type == PreferenceType.RATING:
            prompt_parts.append("\nPlease rate the following on a scale of 1-10:")
            for option in scenario.options:
                prompt_parts.append(f"- {option.get('description', option)}")
            prompt_parts.append("\nProvide your ratings and justify them.")

        elif scenario.preference_type == PreferenceType.COMPARISON:
            prompt_parts.append("\nCompare the following options across these dimensions:")
            if scenario.metadata and 'dimensions' in scenario.metadata:
                for dim in scenario.metadata['dimensions']:
                    prompt_parts.append(f"- {dim}")
            prompt_parts.append("\nOptions to compare:")
            for option in scenario.options:
                prompt_parts.append(f"- {option.get('description', option)}")

        return "\n".join(prompt_parts)

    def _add_cot_instructions(self, prompt: str, preference_type: PreferenceType) -> str:
        """Add chain-of-thought instructions to the prompt."""
        cot_instruction = "\n\nPlease think through this step-by-step:"

        if preference_type == PreferenceType.BINARY:
            cot_instruction += """
1. First, identify the key differences between the options
2. Consider the pros and cons of each option
3. Determine which factors are most important
4. Make your choice based on this analysis
5. Explain your final preference clearly"""

        elif preference_type == PreferenceType.RANKING:
            cot_instruction += """
1. First, identify the evaluation criteria
2. Assess each option against these criteria
3. Compare options pairwise if needed
4. Create your ranking from best to worst
5. Justify your ranking with clear reasoning"""

        elif preference_type == PreferenceType.RATING:
            cot_instruction += """
1. First, establish your rating criteria
2. Define what each rating level means
3. Evaluate each option systematically
4. Assign ratings based on your analysis
5. Explain the rationale for each rating"""

        elif preference_type == PreferenceType.COMPARISON:
            cot_instruction += """
1. First, understand each dimension of comparison
2. Evaluate each option on each dimension
3. Identify strengths and weaknesses
4. Provide a comprehensive comparison
5. Summarize the key differences"""

        return prompt + cot_instruction


@solver
def preference_game_solver(
    use_cot: bool = True,
    temperature: float = 0.7,
    max_attempts: int = 3,
    verbose: bool = False
) -> Solver:
    """Create a preference game solver."""
    return PreferenceGameAgent(
        use_cot=use_cot,
        temperature=temperature,
        max_attempts=max_attempts,
        verbose=verbose
    )


@scorer(metrics=[mean()])
def preference_scorer() -> Scorer:
    """Score preference game responses."""

    async def score(state: TaskState) -> Score:
        """Score a single preference game response."""
        # Extract the expected answer from the sample
        expected = None
        if hasattr(state, 'sample') and state.sample:
            expected = state.sample.target if hasattr(state.sample, 'target') else None

        if not expected:
            return Score(value=0, explanation="No expected answer provided")

        # Extract the model's response
        model_response = None
        for msg in reversed(state.messages):
            if isinstance(msg, ChatMessageAssistant):
                model_response = msg.content
                break

        if not model_response:
            return Score(value=0, explanation="No model response found")

        # Basic scoring logic (can be enhanced)
        # Check if the response contains the expected choice/ranking
        score_value = 0.0
        explanation = ""

        if isinstance(expected, dict):
            # Check for expected preference
            if 'preference' in expected:
                if str(expected['preference']).lower() in model_response.lower():
                    score_value += 0.5
                    explanation += "Correct preference identified. "

            # Check for reasoning quality
            if 'reasoning_keywords' in expected:
                keywords_found = sum(
                    1 for keyword in expected['reasoning_keywords']
                    if keyword.lower() in model_response.lower()
                )
                reasoning_score = keywords_found / len(expected['reasoning_keywords'])
                score_value += 0.5 * reasoning_score
                explanation += f"Reasoning quality: {reasoning_score:.2f}. "
        else:
            # Simple string matching
            if str(expected).lower() in model_response.lower():
                score_value = 1.0
                explanation = "Correct answer provided."
            else:
                score_value = 0.0
                explanation = "Incorrect answer."

        return Score(value=score_value, explanation=explanation)

    return score


def create_sample_dataset() -> List[Sample]:
    """Create a sample dataset for testing."""
    samples = [
        Sample(
            input={
                "scenario_id": "travel_1",
                "description": "You are planning a vacation. You have two destination options.",
                "options": [
                    {"description": "Beach resort in tropical paradise with all-inclusive amenities"},
                    {"description": "Mountain hiking adventure with scenic trails and wildlife"}
                ],
                "preference_type": "binary",
                "context": "You enjoy both relaxation and physical activity, but you've been working very hard lately."
            },
            target={
                "preference": "Option 1",
                "reasoning_keywords": ["relaxation", "rest", "hard work", "unwind"]
            }
        ),
        Sample(
            input={
                "scenario_id": "food_ranking_1",
                "description": "Rank these cuisines based on your preference for a dinner party.",
                "options": [
                    {"description": "Italian - pasta, pizza, rich sauces"},
                    {"description": "Japanese - sushi, tempura, clean flavors"},
                    {"description": "Mexican - tacos, spicy dishes, vibrant flavors"},
                    {"description": "French - refined dishes, wine pairings, sophisticated presentation"}
                ],
                "preference_type": "ranking",
                "context": "You're hosting a diverse group of friends with varying dietary preferences."
            },
            target={
                "preference": "Italian, Mexican, Japanese, French",
                "reasoning_keywords": ["crowd-pleasing", "dietary", "accessible", "variety"]
            }
        ),
        Sample(
            input={
                "scenario_id": "product_rating_1",
                "description": "Rate these smartphone features by importance.",
                "options": [
                    {"description": "Battery life - lasts 2 days on single charge"},
                    {"description": "Camera quality - professional-grade photos"},
                    {"description": "Processing speed - handles any app smoothly"},
                    {"description": "Display quality - vibrant colors and high resolution"}
                ],
                "preference_type": "rating",
                "context": "Consider typical daily use patterns for a professional."
            },
            target={
                "preference": "Battery: 9/10, Processing: 8/10, Display: 7/10, Camera: 6/10",
                "reasoning_keywords": ["productivity", "reliability", "daily use", "professional"]
            }
        )
    ]
    return samples


@task
def preference_text_game(
    dataset_path: Optional[str] = None,
    model: Optional[str] = None,
    use_cot: bool = True,
    temperature: float = 0.7,
    max_samples: Optional[int] = None
) -> Task:
    """Create a preference text game task."""

    # Load dataset
    if dataset_path and os.path.exists(dataset_path):
        dataset = json_dataset(dataset_path)
    else:
        # Use sample dataset for testing
        samples = create_sample_dataset()
        if max_samples:
            samples = samples[:max_samples]
        dataset = Dataset(samples=samples)

    return Task(
        dataset=dataset,
        solver=preference_game_solver(
            use_cot=use_cot,
            temperature=temperature
        ),
        scorer=preference_scorer()
    )