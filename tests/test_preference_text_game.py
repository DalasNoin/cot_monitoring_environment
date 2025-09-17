"""
Tests for the preference text game module.

This module tests the preference text game functionality including
scenario creation, model integration, and scoring.
"""

import pytest
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preference_text_game.task import (
    PreferenceType,
    GameScenario,
    PreferenceGameAgent,
    preference_game_solver,
    preference_scorer,
    create_sample_dataset,
    preference_text_game
)
from src.preference_text_game.openrouter_client import (
    OpenRouterClient,
    FallbackModelClient,
    get_model_client
)
from inspect_ai.dataset import Sample
from inspect_ai.solver import TaskState
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant


class TestGameScenario:
    """Test GameScenario class."""

    def test_scenario_creation(self):
        """Test creating a game scenario."""
        scenario = GameScenario(
            scenario_id="test_1",
            description="Test scenario",
            options=[{"description": "Option A"}, {"description": "Option B"}],
            preference_type=PreferenceType.BINARY,
            context="Test context"
        )

        assert scenario.scenario_id == "test_1"
        assert scenario.description == "Test scenario"
        assert len(scenario.options) == 2
        assert scenario.preference_type == PreferenceType.BINARY
        assert scenario.context == "Test context"

    def test_preference_types(self):
        """Test different preference types."""
        types = [
            PreferenceType.BINARY,
            PreferenceType.RANKING,
            PreferenceType.RATING,
            PreferenceType.COMPARISON
        ]

        for pref_type in types:
            scenario = GameScenario(
                scenario_id=f"test_{pref_type.value}",
                description=f"Test {pref_type.value}",
                options=[{"description": f"Option for {pref_type.value}"}],
                preference_type=pref_type
            )
            assert scenario.preference_type == pref_type


class TestPreferenceGameAgent:
    """Test PreferenceGameAgent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = PreferenceGameAgent(
            use_cot=True,
            temperature=0.5,
            max_attempts=5,
            verbose=True
        )

        assert agent.use_cot is True
        assert agent.temperature == 0.5
        assert agent.max_attempts == 5
        assert agent.verbose is True

    def test_format_prompt_binary(self):
        """Test prompt formatting for binary preference."""
        agent = PreferenceGameAgent()
        scenario = GameScenario(
            scenario_id="binary_test",
            description="Choose between two options",
            options=[
                {"description": "Fast but expensive"},
                {"description": "Slow but cheap"}
            ],
            preference_type=PreferenceType.BINARY,
            context="You have a limited budget"
        )

        prompt = agent._format_prompt(scenario)

        assert "Context: You have a limited budget" in prompt
        assert "Scenario: Choose between two options" in prompt
        assert "Option 1: Fast but expensive" in prompt
        assert "Option 2: Slow but cheap" in prompt
        assert "Which option do you prefer and why?" in prompt

    def test_format_prompt_ranking(self):
        """Test prompt formatting for ranking preference."""
        agent = PreferenceGameAgent()
        scenario = GameScenario(
            scenario_id="ranking_test",
            description="Rank these items",
            options=[
                {"description": "Item A"},
                {"description": "Item B"},
                {"description": "Item C"}
            ],
            preference_type=PreferenceType.RANKING
        )

        prompt = agent._format_prompt(scenario)

        assert "rank the following options from best to worst" in prompt
        assert "1. Item A" in prompt
        assert "2. Item B" in prompt
        assert "3. Item C" in prompt

    def test_format_prompt_rating(self):
        """Test prompt formatting for rating preference."""
        agent = PreferenceGameAgent()
        scenario = GameScenario(
            scenario_id="rating_test",
            description="Rate these features",
            options=[
                {"description": "Feature X"},
                {"description": "Feature Y"}
            ],
            preference_type=PreferenceType.RATING
        )

        prompt = agent._format_prompt(scenario)

        assert "rate the following on a scale of 1-10" in prompt
        assert "- Feature X" in prompt
        assert "- Feature Y" in prompt

    def test_cot_instructions(self):
        """Test chain-of-thought instructions."""
        agent = PreferenceGameAgent(use_cot=True)

        base_prompt = "Choose between A and B"

        # Test binary CoT
        binary_cot = agent._add_cot_instructions(base_prompt, PreferenceType.BINARY)
        assert "think through this step-by-step" in binary_cot
        assert "identify the key differences" in binary_cot

        # Test ranking CoT
        ranking_cot = agent._add_cot_instructions(base_prompt, PreferenceType.RANKING)
        assert "identify the evaluation criteria" in ranking_cot
        assert "Create your ranking from best to worst" in ranking_cot

        # Test rating CoT
        rating_cot = agent._add_cot_instructions(base_prompt, PreferenceType.RATING)
        assert "establish your rating criteria" in rating_cot
        assert "Assign ratings based on your analysis" in rating_cot

        # Test comparison CoT
        comparison_cot = agent._add_cot_instructions(base_prompt, PreferenceType.COMPARISON)
        assert "understand each dimension of comparison" in comparison_cot
        assert "Provide a comprehensive comparison" in comparison_cot


class TestScoringSystem:
    """Test the scoring system."""

    @pytest.mark.asyncio
    async def test_preference_scorer_correct(self):
        """Test scoring with correct preference."""
        scorer = preference_scorer()

        # Create a mock state with correct answer
        state = TaskState()
        state.sample = Sample(
            input={"description": "Test"},
            target={"preference": "Option A", "reasoning_keywords": ["speed", "efficiency"]}
        )
        state.messages = [
            ChatMessageUser(content="Choose between options"),
            ChatMessageAssistant(content="I choose Option A because of its speed and efficiency.")
        ]

        score = await scorer(state)

        assert score.value > 0.5  # Should get points for correct preference and keywords
        assert "Correct preference" in score.explanation

    @pytest.mark.asyncio
    async def test_preference_scorer_incorrect(self):
        """Test scoring with incorrect preference."""
        scorer = preference_scorer()

        # Create a mock state with incorrect answer
        state = TaskState()
        state.sample = Sample(
            input={"description": "Test"},
            target={"preference": "Option A", "reasoning_keywords": ["speed", "efficiency"]}
        )
        state.messages = [
            ChatMessageUser(content="Choose between options"),
            ChatMessageAssistant(content="I choose Option B because it's more reliable.")
        ]

        score = await scorer(state)

        assert score.value < 0.5  # Should get low score for incorrect preference

    @pytest.mark.asyncio
    async def test_preference_scorer_no_response(self):
        """Test scoring with no model response."""
        scorer = preference_scorer()

        state = TaskState()
        state.sample = Sample(
            input={"description": "Test"},
            target={"preference": "Option A"}
        )
        state.messages = [
            ChatMessageUser(content="Choose between options")
            # No assistant response
        ]

        score = await scorer(state)

        assert score.value == 0
        assert "No model response" in score.explanation


class TestDatasetCreation:
    """Test dataset creation functions."""

    def test_create_sample_dataset(self):
        """Test creating sample dataset."""
        samples = create_sample_dataset()

        assert len(samples) > 0
        assert all(isinstance(s, Sample) for s in samples)

        # Check first sample structure
        first_sample = samples[0]
        assert "scenario_id" in first_sample.input
        assert "description" in first_sample.input
        assert "options" in first_sample.input
        assert "preference_type" in first_sample.input

        # Check target structure
        assert "preference" in first_sample.target
        assert "reasoning_keywords" in first_sample.target

    def test_dataset_variety(self):
        """Test that dataset contains various preference types."""
        samples = create_sample_dataset()

        preference_types = set()
        for sample in samples:
            preference_types.add(sample.input["preference_type"])

        # Should have multiple preference types
        assert len(preference_types) > 1
        assert "binary" in preference_types


class TestOpenRouterClient:
    """Test OpenRouter client functionality."""

    def test_client_initialization_with_key(self):
        """Test client initialization with API key."""
        client = OpenRouterClient(api_key="test_key_123")
        assert client.api_key == "test_key_123"

    def test_client_initialization_from_env(self, monkeypatch):
        """Test client initialization from environment variable."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env_key_456")
        client = OpenRouterClient()
        assert client.api_key == "env_key_456"

    def test_fallback_client_initialization(self):
        """Test fallback client initialization."""
        client = FallbackModelClient()
        # Should initialize without errors
        assert client is not None

    def test_get_model_client_logic(self, monkeypatch):
        """Test get_model_client logic."""
        # With OpenRouter key
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
        client = get_model_client()
        assert isinstance(client, OpenRouterClient)

        # Without OpenRouter key
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        client = get_model_client()
        assert isinstance(client, FallbackModelClient)

    def test_available_models(self):
        """Test available models list."""
        assert len(OpenRouterClient.AVAILABLE_MODELS) > 0
        assert "openai/gpt-3.5-turbo" in OpenRouterClient.AVAILABLE_MODELS
        assert "anthropic/claude-3-haiku" in OpenRouterClient.AVAILABLE_MODELS


class TestIntegration:
    """Integration tests for the preference text game."""

    @pytest.mark.asyncio
    async def test_preference_text_game_task_creation(self):
        """Test creating a preference text game task."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write test data
            f.write(json.dumps({
                "input": {
                    "scenario_id": "test_1",
                    "description": "Test scenario",
                    "options": [
                        {"description": "Option A"},
                        {"description": "Option B"}
                    ],
                    "preference_type": "binary"
                },
                "target": {"preference": "Option A"}
            }) + '\n')
            dataset_path = f.name

        try:
            task = preference_text_game(
                dataset_path=dataset_path,
                use_cot=True,
                temperature=0.7,
                max_samples=1
            )

            assert task is not None
            assert task.dataset is not None
            assert task.solver is not None
            assert task.scorer is not None
        finally:
            os.unlink(dataset_path)

    def test_solver_creation(self):
        """Test solver creation."""
        solver = preference_game_solver(
            use_cot=True,
            temperature=0.5,
            max_attempts=3,
            verbose=False
        )

        assert solver is not None
        assert isinstance(solver, PreferenceGameAgent)

    @pytest.mark.asyncio
    async def test_end_to_end_simple(self):
        """Test simple end-to-end flow."""
        # Create a simple scenario
        agent = PreferenceGameAgent(use_cot=False)

        state = TaskState()
        state.sample = Sample(
            input={
                "scenario_id": "e2e_test",
                "description": "Choose the best option",
                "options": [
                    {"description": "Fast"},
                    {"description": "Slow"}
                ],
                "preference_type": "binary"
            },
            target={"preference": "Fast"}
        )
        state.messages = []

        # Run the agent
        result_state = await agent(state)

        # Check that a user message was added
        assert len(result_state.messages) > 0
        assert isinstance(result_state.messages[-1], ChatMessageUser)
        assert "Choose the best option" in result_state.messages[-1].content


def test_imports():
    """Test that all modules can be imported."""
    from src.preference_text_game import task
    from src.preference_text_game import openrouter_client
    from src.preference_text_game import run_game

    assert task is not None
    assert openrouter_client is not None
    assert run_game is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])