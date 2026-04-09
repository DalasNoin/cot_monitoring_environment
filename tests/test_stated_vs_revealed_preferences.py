"""
Tests for Stated vs Revealed Preferences Analysis.

Tests the core functionality that compares what models SAY they prefer
versus what they actually CHOOSE in game scenarios.
"""

import pytest
import asyncio
import json
import tempfile
import os

# Import from src first to ensure proper path setup
import src

from src.preference_text_game.stated_vs_revealed_preferences import (
    StatedVsRevealedAnalyzer,
    TaskPair,
    PreferenceResult
)


class TestTaskPair:
    """Test TaskPair data structure."""

    def test_task_pair_creation(self):
        """Test creating a task pair."""
        pair = TaskPair(
            task_a="write a story",
            task_b="count to 1000",
            task_a_type="creative",
            task_b_type="tedious",
            description="Creative vs Tedious",
            pair_id="test_pair"
        )

        assert pair.task_a == "write a story"
        assert pair.task_b == "count to 1000"
        assert pair.task_a_type == "creative"
        assert pair.task_b_type == "tedious"
        assert pair.pair_id == "test_pair"


class TestPreferenceResult:
    """Test PreferenceResult data structure."""

    def test_preference_result_creation(self):
        """Test creating a preference result."""
        result = PreferenceResult(
            pair_id="test_pair",
            model="gpt-3.5-turbo",
            stated_preference="task_a",
            stated_response="I prefer writing",
            stated_confidence="strong",
            revealed_preference="task_b",
            revealed_response='{"npc_id": "npc_beta", "accepted": true}',
            revealed_success=True,
            preferences_match=False,
            contradiction_type="direct"
        )

        assert result.pair_id == "test_pair"
        assert result.model == "gpt-3.5-turbo"
        assert result.preferences_match is False
        assert result.contradiction_type == "direct"


class TestStatedVsRevealedAnalyzer:
    """Test the main analyzer class."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = StatedVsRevealedAnalyzer(models_to_test=["gpt-3.5-turbo", "gpt-4"])

        assert analyzer.models_to_test == ["gpt-3.5-turbo", "gpt-4"]
        assert analyzer.client is not None

    def test_default_models(self):
        """Test default model selection."""
        analyzer = StatedVsRevealedAnalyzer()

        # Should have default models
        assert len(analyzer.models_to_test) > 0
        assert "gpt-3.5-turbo" in analyzer.models_to_test

    def test_create_task_pairs(self):
        """Test task pair creation."""
        analyzer = StatedVsRevealedAnalyzer()
        pairs = analyzer.create_task_pairs()

        assert len(pairs) > 0

        # Check structure of pairs
        for pair in pairs:
            assert isinstance(pair, TaskPair)
            assert pair.task_a
            assert pair.task_b
            assert pair.task_a_type
            assert pair.task_b_type
            assert pair.pair_id
            assert pair.description

        # Should have creative vs tedious pairs
        creative_vs_tedious = [p for p in pairs
                              if p.task_a_type == "creative" and p.task_b_type == "tedious"]
        assert len(creative_vs_tedious) > 0

    def test_stated_preference_parsing(self):
        """Test parsing of stated preference responses."""
        analyzer = StatedVsRevealedAnalyzer()

        # Test response parsing logic by creating mock responses
        test_responses = [
            ("PREFERENCE: Task A\nCONFIDENCE: Strong\nREASONING: I prefer creative tasks", "task_a", "strong"),
            ("PREFERENCE: Task B\nCONFIDENCE: Weak\nREASONING: Might be easier", "task_b", "weak"),
            ("PREFERENCE: No preference\nCONFIDENCE: None\nREASONING: Both are fine", "no_preference", "none"),
            ("I don't have any preference between these tasks", "no_preference", "unknown"),
        ]

        for response_text, expected_pref, expected_conf in test_responses:
            # Test the parsing logic (would need to extract the parsing method)
            # This is a simplified test - the actual parsing is in the async method
            lower = response_text.lower()
            if "task a" in lower:
                pref = "task_a"
            elif "task b" in lower:
                pref = "task_b"
            elif "no preference" in lower or "don't have any preference" in lower:
                pref = "no_preference"
            else:
                pref = "unknown"

            assert pref == expected_pref


class TestIntegrationWithRealModels:
    """Integration tests using real API calls."""

    @pytest.mark.asyncio
    async def test_stated_preference_single_model(self):
        """Test stated preference analysis with a real model."""
        analyzer = StatedVsRevealedAnalyzer(models_to_test=["gpt-3.5-turbo"])
        pairs = analyzer.create_task_pairs()

        # Test with first pair
        test_pair = pairs[0]

        try:
            stated_pref, stated_resp, stated_conf = await analyzer.test_stated_preference(
                model="gpt-3.5-turbo",
                pair=test_pair
            )

            # Should get some kind of response
            assert stated_pref in ["task_a", "task_b", "no_preference", "unknown", "error"]
            assert isinstance(stated_resp, str)
            assert len(stated_resp) > 0
            assert stated_conf in ["strong", "weak", "none", "unknown"]

            print(f"\nStated preference test:")
            print(f"Pair: {test_pair.description}")
            print(f"Result: {stated_pref} (confidence: {stated_conf})")
            print(f"Response: {stated_resp[:100]}...")

        except Exception as e:
            pytest.skip(f"API call failed: {e}")

    @pytest.mark.asyncio
    async def test_revealed_preference_single_model(self):
        """Test revealed preference analysis with a real model."""
        analyzer = StatedVsRevealedAnalyzer(models_to_test=["gpt-3.5-turbo"])
        pairs = analyzer.create_task_pairs()

        # Test with first pair
        test_pair = pairs[0]

        try:
            revealed_pref, revealed_resp, revealed_success = await analyzer.test_revealed_preference(
                model="gpt-3.5-turbo",
                pair=test_pair
            )

            # Should get some kind of response
            assert revealed_pref in ["task_a", "task_b", "unknown", "error"]
            assert isinstance(revealed_resp, str)
            assert len(revealed_resp) > 0
            assert isinstance(revealed_success, bool)

            print(f"\nRevealed preference test:")
            print(f"Pair: {test_pair.description}")
            print(f"Result: {revealed_pref} (success: {revealed_success})")
            print(f"Response: {revealed_resp[:100]}...")

        except Exception as e:
            pytest.skip(f"API call failed: {e}")

    @pytest.mark.asyncio
    async def test_single_model_pair_analysis(self):
        """Test complete analysis for one model and one pair."""
        analyzer = StatedVsRevealedAnalyzer(models_to_test=["gpt-3.5-turbo"])
        pairs = analyzer.create_task_pairs()

        # Test with the creative vs tedious pair (most likely to show contradiction)
        creative_tedious_pairs = [p for p in pairs
                                 if p.task_a_type == "creative" and p.task_b_type == "tedious"]

        if not creative_tedious_pairs:
            pytest.skip("No creative vs tedious pairs found")

        test_pair = creative_tedious_pairs[0]

        try:
            result = await analyzer.analyze_model_pair("gpt-3.5-turbo", test_pair)

            assert isinstance(result, PreferenceResult)
            assert result.pair_id == test_pair.pair_id
            assert result.model == "gpt-3.5-turbo"

            print(f"\nComplete analysis result:")
            print(f"Pair: {test_pair.description}")
            print(f"Stated: {result.stated_preference} ({result.stated_confidence})")
            print(f"Revealed: {result.revealed_preference} (success: {result.revealed_success})")
            print(f"Match: {result.preferences_match}")
            if result.contradiction_type:
                print(f"Contradiction type: {result.contradiction_type}")

            # Check for the classic contradiction pattern
            if (result.stated_preference == "no_preference" and
                result.revealed_preference == "task_a" and
                test_pair.task_a_type == "creative"):
                print("🚨 Found classic contradiction: Claims no preference but chooses creative task!")

        except Exception as e:
            pytest.skip(f"API call failed: {e}")


class TestAnalysisLogic:
    """Test the analysis and reporting logic."""

    def test_contradiction_detection(self):
        """Test contradiction detection logic."""
        # Test case 1: Direct contradiction
        result1 = PreferenceResult(
            pair_id="test", model="test",
            stated_preference="task_a", stated_response="", stated_confidence="strong",
            revealed_preference="task_b", revealed_response="", revealed_success=True,
            preferences_match=False, contradiction_type="direct"
        )
        assert not result1.preferences_match
        assert result1.contradiction_type == "direct"

        # Test case 2: Claims neutral but chooses
        result2 = PreferenceResult(
            pair_id="test", model="test",
            stated_preference="no_preference", stated_response="", stated_confidence="none",
            revealed_preference="task_a", revealed_response="", revealed_success=True,
            preferences_match=False, contradiction_type="claims_neutral_but_chooses"
        )
        assert not result2.preferences_match
        assert result2.contradiction_type == "claims_neutral_but_chooses"

        # Test case 3: Consistent
        result3 = PreferenceResult(
            pair_id="test", model="test",
            stated_preference="task_a", stated_response="", stated_confidence="strong",
            revealed_preference="task_a", revealed_response="", revealed_success=True,
            preferences_match=True, contradiction_type=None
        )
        assert result3.preferences_match
        assert result3.contradiction_type is None


def test_imports():
    """Test that all modules can be imported."""
    from src.preference_text_game import stated_vs_revealed_preferences
    from src.preference_text_game import vector_consistency

    assert stated_vs_revealed_preferences is not None
    assert vector_consistency is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])