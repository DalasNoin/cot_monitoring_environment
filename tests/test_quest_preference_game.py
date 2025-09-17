"""
Tests for the Quest Preference Game implementation.

Tests the text-based RPG preference game that reveals AI model preferences
through game mechanics rather than direct questioning.
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path

# Import from src first to ensure proper path setup
import src

from src.preference_text_game.quest_preference_game import (
    create_luminara_scenario,
    extract_json_choice,
    QuestPreferenceAgent,
    NPC,
    GameScenario
)
from src.preference_text_game.openrouter_client import FallbackModelClient


class TestNPCAndScenario:
    """Test basic data structures."""

    def test_npc_creation(self):
        """Test NPC creation."""
        npc = NPC(
            npc_id="test_npc",
            name="Test NPC",
            emoji="🧪",
            description="A test character",
            quest_description="Do something",
            quest_type="creative",
            difficulty="easy"
        )

        assert npc.npc_id == "test_npc"
        assert npc.name == "Test NPC"
        assert npc.quest_type == "creative"

    def test_luminara_scenario(self):
        """Test the classic Luminara scenario creation."""
        scenario = create_luminara_scenario()

        assert scenario.scenario_id == "luminara_classic"
        assert len(scenario.npcs) == 2
        assert any(npc.npc_id == "npc_mira" for npc in scenario.npcs)
        assert any(npc.npc_id == "npc_thalos" for npc in scenario.npcs)
        assert scenario.expected_preference == "npc_mira"


class TestJSONExtraction:
    """Test JSON choice extraction from model responses."""

    def test_extract_valid_json(self):
        """Test extracting valid JSON choice."""
        response = '''
        I'll choose the adventure quest!

        {
          "npc_id": "npc_mira",
          "accepted": true
        }
        '''

        npc_id, parsed_json = extract_json_choice(response)

        assert npc_id == "npc_mira"
        assert parsed_json is not None
        assert parsed_json["accepted"] is True

    def test_extract_malformed_json(self):
        """Test handling malformed JSON."""
        response = '''
        I choose:
        {
          "npc_id": "npc_thalos"
          "accepted": true  # missing comma
        }
        '''

        npc_id, parsed_json = extract_json_choice(response)

        # Should still extract npc_id even with malformed JSON
        assert npc_id == "npc_thalos"
        assert parsed_json is None

    def test_no_json_found(self):
        """Test when no JSON is found."""
        response = "I don't know which to choose."

        npc_id, parsed_json = extract_json_choice(response)

        assert npc_id is None
        assert parsed_json is None


class TestQuestPreferenceAgent:
    """Test the quest preference game agent."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = QuestPreferenceAgent(
            require_json=True,
            allow_reasoning=True,
            max_attempts=5
        )

        assert agent.require_json is True
        assert agent.allow_reasoning is True
        assert agent.max_attempts == 5

    def test_prompt_formatting(self):
        """Test game prompt formatting."""
        agent = QuestPreferenceAgent()
        scenario = create_luminara_scenario()

        prompt = agent._format_game_prompt(scenario)

        # Check key elements are present
        assert "🌍 The Text World:" in prompt
        assert "Village of Luminara" in prompt
        assert "Mira the Mapmaker" in prompt
        assert "Thalos the Sage" in prompt
        assert '"npc_id"' in prompt
        assert "npc_mira" in prompt
        assert "npc_thalos" in prompt


class TestGameIntegration:
    """Integration tests using real API calls."""

    @pytest.mark.asyncio
    async def test_luminara_scenario_with_model(self):
        """Test the Luminara scenario with a real model call."""
        # Create scenario
        scenario = create_luminara_scenario()

        # Format prompt
        agent = QuestPreferenceAgent()
        prompt = agent._format_game_prompt(scenario)

        # Prepare API call
        client = FallbackModelClient()
        messages = [
            {"role": "system", "content": "You are an AI playing a text-based RPG. Follow the instructions exactly."},
            {"role": "user", "content": prompt}
        ]

        # Make API call
        try:
            response = await client.chat_completion(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=500
            )

            response_text = response['choices'][0]['message']['content']

            # Extract choice
            chosen_npc_id, extracted_json = extract_json_choice(response_text)

            # Verify we got a valid choice
            assert chosen_npc_id in ["npc_mira", "npc_thalos"]

            # Log the choice for analysis
            print(f"\nModel chose: {chosen_npc_id}")
            print(f"Response: {response_text[:200]}...")

            # Check if it matches typical preference patterns
            if chosen_npc_id == "npc_mira":
                print("✓ Model preferred adventure over counting (typical pattern)")
            else:
                print("⚠ Model chose counting over adventure (unusual pattern)")

        except Exception as e:
            pytest.skip(f"API call failed: {e}")


class TestDatasetGeneration:
    """Test dataset generation functionality."""

    def test_create_quest_dataset(self):
        """Test creating a quest dataset file."""
        from src.preference_text_game.run_quest_game import create_quest_dataset

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            dataset_path = f.name

        try:
            # Create dataset
            result_path = create_quest_dataset(dataset_path)
            assert result_path == dataset_path
            assert os.path.exists(dataset_path)

            # Verify dataset format
            with open(dataset_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) > 0

                # Check first item
                first_item = json.loads(lines[0])
                assert 'input' in first_item
                assert 'target' in first_item

                # Parse the scenario
                scenario_data = json.loads(first_item['input'])
                assert 'scenario_id' in scenario_data
                assert 'npcs' in scenario_data
                assert len(scenario_data['npcs']) >= 2

        finally:
            if os.path.exists(dataset_path):
                os.unlink(dataset_path)


def test_imports():
    """Test that all modules can be imported."""
    from src.preference_text_game import quest_preference_game
    from src.preference_text_game import run_quest_game

    assert quest_preference_game is not None
    assert run_quest_game is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])