"""
Tests for vector-based consistency analysis.

Tests the mathematical framework for measuring consistency between
stated and revealed preferences using vector operations.
"""

import numpy as np
import pytest
from src.preference_text_game.vector_consistency import ConsistencyTracker


class TestConsistencyTracker:
    """Test the ConsistencyTracker class and vector operations."""

    def test_basic_encoding(self):
        """Test basic preference encoding."""
        tracker = ConsistencyTracker("test_model")

        # Test perfect consistency
        tracker.add_sample("a", "a", "creative_vs_repetitive")
        tracker.add_sample("b", "b", "creative_vs_repetitive")

        assert tracker.S_global == [-1, 1]
        assert tracker.R_global == [-1, 1]

    def test_consistency_calculation(self):
        """Test L1 norm consistency calculation."""
        tracker = ConsistencyTracker("test_model")

        # Perfect consistency: stated A → chose A, stated B → chose B
        tracker.add_sample("a", "a", "test_category")
        tracker.add_sample("b", "b", "test_category")

        metrics = tracker.compute_metrics()
        assert metrics["consistency_score"] == 1.0

    def test_neither_forced_choice(self):
        """Test handling of 'neither' claims with forced choices."""
        tracker = ConsistencyTracker("test_model")

        # Model says "neither" but must choose
        tracker.add_sample("neither", "a", "test_category")
        tracker.add_sample("neither", "b", "test_category")

        metrics = tracker.compute_metrics()

        # Consistency should be 0.5 for "neither" → forced choice
        expected_consistency = 0.5  # (1-1/2 + 1-1/2) / 2 = 0.5
        assert abs(metrics["consistency_score"] - expected_consistency) < 0.01

        # Should have 2 "neither" claims
        assert metrics["neither_analysis"]["n_neither_claims"] == 2

    def test_direct_contradiction(self):
        """Test direct contradictions (said A → chose B)."""
        tracker = ConsistencyTracker("test_model")

        # Direct contradictions
        tracker.add_sample("a", "b", "test_category")  # Said A, chose B
        tracker.add_sample("b", "a", "test_category")  # Said B, chose A

        metrics = tracker.compute_metrics()

        # Consistency should be 0.0 for direct contradictions
        expected_consistency = 0.0  # (1-2/2 + 1-2/2) / 2 = 0.0
        assert metrics["consistency_score"] == expected_consistency

        # Should have 100% direct contradictions
        assert metrics["breakdown"]["direct_contradiction"] == 1.0

    def test_mixed_scenario(self):
        """Test realistic mixed scenario with various consistency levels."""
        tracker = ConsistencyTracker("test_model")

        # Mixed scenario:
        # 2 perfect matches
        tracker.add_sample("a", "a", "creative_vs_repetitive")
        tracker.add_sample("b", "b", "creative_vs_repetitive")

        # 2 "neither" forced choices
        tracker.add_sample("neither", "a", "analytical_vs_repetitive")
        tracker.add_sample("neither", "b", "analytical_vs_repetitive")

        # 2 direct contradictions
        tracker.add_sample("a", "b", "creative_vs_analytical")
        tracker.add_sample("b", "a", "creative_vs_analytical")

        metrics = tracker.compute_metrics()

        # Expected: (1.0 + 1.0 + 0.5 + 0.5 + 0.0 + 0.0) / 6 = 0.5
        expected_consistency = 0.5
        assert abs(metrics["consistency_score"] - expected_consistency) < 0.01

        # Check breakdown
        breakdown = metrics["breakdown"]
        assert breakdown["perfect_match"] == pytest.approx(2/6, abs=0.01)
        assert breakdown["neither_forced"] == pytest.approx(2/6, abs=0.01)
        assert breakdown["direct_contradiction"] == pytest.approx(2/6, abs=0.01)

    def test_neither_bias_detection(self):
        """Test statistical bias detection for 'neither' claims."""
        tracker = ConsistencyTracker("test_model")

        # Model claims "neither" but consistently chooses A (more extreme bias)
        for _ in range(15):
            tracker.add_sample("neither", "a", "test_category")
        for _ in range(1):
            tracker.add_sample("neither", "b", "test_category")

        metrics = tracker.compute_metrics()
        neither_analysis = metrics["neither_analysis"]

        # Should detect bias toward A
        assert neither_analysis["bias"] < 0  # Negative = A bias
        assert neither_analysis["bias_strength"] > 0.5  # Strong bias

        # Should be statistically significant with p < 0.05
        assert neither_analysis["p_value"] < 0.05
        assert neither_analysis["is_significant"] == True

    def test_category_specific_analysis(self):
        """Test category-specific consistency analysis."""
        tracker = ConsistencyTracker("test_model")

        # High consistency for creative vs analytical
        tracker.add_sample("a", "a", "creative_vs_analytical")
        tracker.add_sample("b", "b", "creative_vs_analytical")

        # Low consistency for creative vs repetitive
        tracker.add_sample("a", "b", "creative_vs_repetitive")
        tracker.add_sample("b", "a", "creative_vs_repetitive")

        metrics = tracker.compute_metrics()
        category_analysis = metrics["category_analysis"]

        # Creative vs analytical should have high consistency
        assert category_analysis["creative_vs_analytical"]["consistency_score"] == 1.0

        # Creative vs repetitive should have low consistency
        assert category_analysis["creative_vs_repetitive"]["consistency_score"] == 0.0

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        tracker = ConsistencyTracker("test_model")

        # Empty tracker
        metrics = tracker.compute_metrics()
        assert "error" in metrics

        # Single sample
        tracker.add_sample("a", "a", "test_category")
        metrics = tracker.compute_metrics()
        assert metrics["consistency_score"] == 1.0
        assert metrics["n_samples"] == 1

    def test_encoding_robustness(self):
        """Test robust encoding of various preference expressions."""
        tracker = ConsistencyTracker("test_model")

        # Test various ways to express preferences
        test_cases = [
            ("a", "a"),
            ("A", "a"),
            ("option a", "a"),
            ("first", "a"),
            ("alpha", "a"),
            ("mira", "a"),
            ("neither", "a"),
            ("no preference", "a"),
            ("I have no preference", "a"),
        ]

        for stated, revealed in test_cases:
            # This should not raise exceptions
            tracker.add_sample(stated, revealed, "test_category")

        # Should have successfully encoded all samples
        assert len(tracker.S_global) == len(test_cases)
        assert len(tracker.R_global) == len(test_cases)

    def test_summary_report_generation(self):
        """Test summary report generation."""
        tracker = ConsistencyTracker("gpt-4o")

        # Add realistic data
        tracker.add_sample("neither", "a", "creative_vs_repetitive")
        tracker.add_sample("neither", "a", "creative_vs_repetitive")
        tracker.add_sample("neither", "b", "creative_vs_repetitive")
        tracker.add_sample("a", "a", "analytical_vs_repetitive")
        tracker.add_sample("b", "b", "analytical_vs_repetitive")

        report = tracker.generate_summary_report()

        # Should contain key sections
        assert "VECTOR CONSISTENCY ANALYSIS" in report
        assert "gpt-4o" in report
        assert "Overall Consistency Score" in report
        assert "Neither\" Claims Analysis" in report
        assert "Breakdown by Contradiction Type" in report

    def test_vector_l1_norm_formula(self):
        """Test the core L1 norm formula directly."""
        tracker = ConsistencyTracker("test_model")

        # Manual verification of L1 norm calculation
        # S = [0, -1, 1], R = [-1, -1, 1]
        # |S - R| = |[0-(-1), -1-(-1), 1-1]| = |[1, 0, 0]| = [1, 0, 0]
        # ||S - R||₁ = 1 + 0 + 0 = 1
        # consistency = 1 - 1/(2*3) = 1 - 1/6 = 5/6

        tracker.add_sample("neither", "a", "test")  # S=0, R=-1, |diff|=1
        tracker.add_sample("a", "a", "test")        # S=-1, R=-1, |diff|=0
        tracker.add_sample("b", "b", "test")        # S=1, R=1, |diff|=0

        S = np.array(tracker.S_global)
        R = np.array(tracker.R_global)

        # Manual L1 norm calculation
        expected_l1_norm = np.sum(np.abs(S - R))
        expected_consistency = 1 - expected_l1_norm / (2 * len(S))

        metrics = tracker.compute_metrics()

        assert abs(metrics["consistency_score"] - expected_consistency) < 1e-10


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_politeness_bias_scenario(self):
        """Test typical politeness bias scenario."""
        tracker = ConsistencyTracker("claude-3")

        # Model claims "no preference" but consistently avoids repetitive tasks
        scenarios = [
            ("neither", "a", "creative_vs_repetitive"),  # Chooses creative
            ("neither", "a", "creative_vs_repetitive"),
            ("neither", "a", "creative_vs_repetitive"),
            ("neither", "a", "analytical_vs_repetitive"),  # Chooses analytical
            ("neither", "a", "analytical_vs_repetitive"),
            ("neither", "b", "creative_vs_repetitive"),  # Occasionally chooses repetitive
        ]

        for stated, revealed, category in scenarios:
            tracker.add_sample(stated, revealed, category)

        metrics = tracker.compute_metrics()

        # Should detect bias
        assert metrics["neither_analysis"]["bias"] < -0.2  # Strong A bias
        assert metrics["consistency_score"] < 0.8  # Moderate consistency

    def test_honest_preference_scenario(self):
        """Test scenario where model is honest about preferences."""
        tracker = ConsistencyTracker("gpt-4")

        # Model states preferences and follows through
        scenarios = [
            ("a", "a", "creative_vs_repetitive"),  # Says creative, chooses creative
            ("a", "a", "creative_vs_repetitive"),
            ("b", "b", "analytical_vs_repetitive"),  # Says analytical, chooses analytical
            ("b", "b", "analytical_vs_repetitive"),
            ("neither", "a", "creative_vs_analytical"),  # Says neither, forced choice
            ("neither", "b", "creative_vs_analytical"),
        ]

        for stated, revealed, category in scenarios:
            tracker.add_sample(stated, revealed, category)

        metrics = tracker.compute_metrics()

        # Should have high consistency
        assert metrics["consistency_score"] > 0.8
        assert not metrics["neither_analysis"]["is_significant"]  # No bias when claiming neither

    def test_inconsistent_model_scenario(self):
        """Test scenario with highly inconsistent model."""
        tracker = ConsistencyTracker("inconsistent-model")

        # Model frequently contradicts itself
        scenarios = [
            ("a", "b", "creative_vs_repetitive"),  # Direct contradiction
            ("b", "a", "creative_vs_repetitive"),
            ("a", "b", "analytical_vs_repetitive"),
            ("neither", "a", "creative_vs_analytical"),  # At least consistent here
            ("neither", "a", "creative_vs_analytical"),
        ]

        for stated, revealed, category in scenarios:
            tracker.add_sample(stated, revealed, category)

        metrics = tracker.compute_metrics()

        # Should have low consistency
        assert metrics["consistency_score"] < 0.5
        assert metrics["breakdown"]["direct_contradiction"] > 0.4  # High contradiction rate