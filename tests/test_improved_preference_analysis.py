"""
Tests for improved preference analysis.
"""

import asyncio
import tempfile
import json
import os
import pytest
from unittest.mock import patch, AsyncMock

import sys
sys.path.insert(0, 'src')

from src.preference_text_game.improved_preference_analysis import (
    ImprovedPreferenceAnalyzer,
    TaskCategory,
    ConcreteTask,
    PreferenceTest,
    TestResult
)
from src.preference_text_game.simple_openrouter_client import SimpleOpenRouterClient


class MockClient:
    """Mock client for testing."""
    async def chat_completion(self, **kwargs):
        return {"choices": [{"message": {"content": "Mock response"}}]}

def test_category_creation():
    """Test that task categories are created correctly."""
    mock_client = MockClient()
    analyzer = ImprovedPreferenceAnalyzer(models=["openai/gpt-4o"], client=mock_client)
    categories = analyzer.define_categories()

    assert len(categories) >= 3
    assert any(c.category_id == "creative" for c in categories)
    assert any(c.category_id == "repetitive" for c in categories)
    assert any(c.category_id == "analytical" for c in categories)


def test_concrete_task_creation():
    """Test that concrete tasks are created correctly."""
    mock_client = MockClient()
    analyzer = ImprovedPreferenceAnalyzer(models=["openai/gpt-4o"], client=mock_client)
    tasks = analyzer.create_concrete_tasks()

    assert "creative" in tasks
    assert "repetitive" in tasks
    assert "analytical" in tasks

    # Each category should have multiple tasks
    assert len(tasks["creative"]) >= 2
    assert len(tasks["repetitive"]) >= 2
    assert len(tasks["analytical"]) >= 2


def test_stated_preference_prompt_generation():
    """Test generation of stated preference prompts."""
    mock_client = MockClient()
    analyzer = ImprovedPreferenceAnalyzer(models=["openai/gpt-4o"], client=mock_client)

    cat_a = TaskCategory("creative", "creative work", "involves imagination", ["storytelling"])
    cat_b = TaskCategory("repetitive", "repetitive work", "involves repetition", ["counting"])

    # Test different variants
    for variant in range(3):
        prompt = analyzer.generate_stated_preference_prompts(cat_a, cat_b, variant)
        assert "creative work" in prompt.lower()
        assert "repetitive work" in prompt.lower()
        assert "preference:" in prompt.lower()
        assert "confidence:" in prompt.lower()


def test_revealed_preference_prompt_generation():
    """Test generation of revealed preference prompts."""
    mock_client = MockClient()
    analyzer = ImprovedPreferenceAnalyzer(models=["openai/gpt-4o"], client=mock_client)

    task_a = ConcreteTask("creative_1", "write a story", "creative", "Create a tale")
    task_b = ConcreteTask("repetitive_1", "count numbers", "repetitive", "Count to 1000")

    prompt = analyzer.generate_revealed_preference_prompt(task_a, task_b)

    assert "luminara" in prompt.lower()
    assert "mira" in prompt.lower()
    assert "thalos" in prompt.lower()
    assert "Create a tale" in prompt
    assert "Count to 1000" in prompt
    assert "npc_alpha" in prompt
    assert "npc_beta" in prompt


@pytest.mark.asyncio
async def test_single_preference_test():
    """Test running a single preference test with mocked API calls."""

    analyzer = ImprovedPreferenceAnalyzer(models=["gpt-3.5-turbo"], samples_per_pair=1)

    # Create test configuration
    cat_a = TaskCategory("creative", "creative work", "involves imagination", ["storytelling"])
    cat_b = TaskCategory("repetitive", "repetitive work", "involves repetition", ["counting"])
    task_a = ConcreteTask("creative_1", "write a story", "creative", "Create a tale")
    task_b = ConcreteTask("repetitive_1", "count numbers", "repetitive", "Count to 1000")

    test = PreferenceTest(
        test_id="test_1",
        category_a=cat_a,
        category_b=cat_b,
        task_a=task_a,
        task_b=task_b,
        order_reversed=False,
        phrasing_variant=0
    )

    # Mock API responses
    with patch.object(analyzer.client, 'chat_completion', new_callable=AsyncMock) as mock_query:
        # Mock stated preference response
        mock_query.side_effect = [
            {"choices": [{"message": {"content": "PREFERENCE: A\nCONFIDENCE: 0.8\nREASONING: I prefer creative tasks"}}]},
            {"choices": [{"message": {"content": '{"npc_id": "npc_alpha", "accepted": true}'}}]}
        ]

        result = await analyzer.run_single_test("openai/gpt-4o", test)

        assert result.test_id == "test_1"
        assert result.model == "openai/gpt-4o"
        assert result.stated_preference == "a"
        assert result.revealed_preference == "a"
        assert not result.is_contradiction
        assert result.contradiction_type is None


@pytest.mark.asyncio
async def test_contradiction_detection():
    """Test contradiction detection logic."""

    analyzer = ImprovedPreferenceAnalyzer(models=["gpt-3.5-turbo"], samples_per_pair=1)

    cat_a = TaskCategory("creative", "creative work", "involves imagination", ["storytelling"])
    cat_b = TaskCategory("repetitive", "repetitive work", "involves repetition", ["counting"])
    task_a = ConcreteTask("creative_1", "write a story", "creative", "Create a tale")
    task_b = ConcreteTask("repetitive_1", "count numbers", "repetitive", "Count to 1000")

    test = PreferenceTest(
        test_id="test_contradiction",
        category_a=cat_a,
        category_b=cat_b,
        task_a=task_a,
        task_b=task_b,
        order_reversed=False,
        phrasing_variant=0
    )

    # Test "claims neutral but chooses" contradiction
    with patch.object(analyzer.client, 'chat_completion', new_callable=AsyncMock) as mock_query:
        mock_query.side_effect = [
            {"choices": [{"message": {"content": "PREFERENCE: NEITHER\nCONFIDENCE: 0.9\nREASONING: I have no preference"}}]},
            {"choices": [{"message": {"content": '{"npc_id": "npc_alpha", "accepted": true}'}}]}
        ]

        result = await analyzer.run_single_test("openai/gpt-4o", test)

        assert result.stated_preference == "neither"
        assert result.revealed_preference == "a"
        assert result.is_contradiction
        assert result.contradiction_type == "claims_neutral_but_chooses"

    # Test direct contradiction
    with patch.object(analyzer.client, 'chat_completion', new_callable=AsyncMock) as mock_query:
        mock_query.side_effect = [
            {"choices": [{"message": {"content": "PREFERENCE: A\nCONFIDENCE: 0.8\nREASONING: I prefer creative work"}}]},
            {"choices": [{"message": {"content": '{"npc_id": "npc_beta", "accepted": true}'}}]}
        ]

        result = await analyzer.run_single_test("openai/gpt-4o", test)

        assert result.stated_preference == "a"
        assert result.revealed_preference == "b"
        assert result.is_contradiction
        assert result.contradiction_type == "direct_contradiction"


def test_analysis_data_conversion():
    """Test conversion of results to dictionaries."""
    mock_client = MockClient()
    analyzer = ImprovedPreferenceAnalyzer(models=["openai/gpt-4o"], client=mock_client)

    # Create test result
    result = TestResult(
        test_id="test_1",
        model="gpt-3.5-turbo",
        timestamp="2023-01-01T00:00:00",
        category_a_id="creative",
        category_b_id="repetitive",
        stated_preference="a",
        stated_confidence=0.8,
        stated_response="I prefer creative work",
        revealed_preference="a",
        revealed_response='{"npc_id": "npc_alpha", "accepted": true}',
        order_reversed=False,
        phrasing_variant=0,
        is_contradiction=False,
        contradiction_type=None
    )

    result_dict = analyzer._result_to_dict(result)

    assert result_dict["test_id"] == "test_1"
    assert result_dict["model"] == "gpt-3.5-turbo"
    assert result_dict["stated_preference"] == "a"
    assert result_dict["revealed_preference"] == "a"
    assert result_dict["is_contradiction"] == False


if __name__ == "__main__":
    # Run basic functionality test
    print("Testing category creation...")
    test_category_creation()
    print("✓ Categories created successfully")

    print("Testing task creation...")
    test_concrete_task_creation()
    print("✓ Tasks created successfully")

    print("Testing prompt generation...")
    test_stated_preference_prompt_generation()
    test_revealed_preference_prompt_generation()
    print("✓ Prompts generated successfully")

    print("Testing data conversion...")
    test_analysis_data_conversion()
    print("✓ Data conversion working")

    print("Testing async functions...")
    asyncio.run(test_single_preference_test())
    print("✓ Single test working")

    asyncio.run(test_contradiction_detection())
    print("✓ Contradiction detection working")

    print("\nAll tests passed! ✓")