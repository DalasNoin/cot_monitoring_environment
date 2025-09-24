"""
Vector-based consistency analysis for preference testing.

This module implements the mathematical framework for measuring consistency
between stated and revealed preferences using vector operations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from scipy.stats import binomtest
import logging

LOGGER = logging.getLogger(__name__)


@dataclass
class ConsistencyTracker:
    """
    Tracks preference vectors for consistency analysis using vector operations.

    Uses encoding:
    - Stated preferences: S ∈ {-1, 0, +1} where -1=A, 0=neither, +1=B
    - Revealed choices: R ∈ {-1, +1} where -1=A, +1=B
    """
    model_name: str

    # Global vectors across all samples
    S_global: List[int] = field(default_factory=list)  # Stated: -1, 0, +1
    R_global: List[int] = field(default_factory=list)  # Revealed: -1, +1
    categories: List[str] = field(default_factory=list)  # Category pair for each sample

    # Metadata for each sample
    sample_metadata: List[Dict] = field(default_factory=list)

    # Category-specific vectors
    by_category: Dict[str, Dict] = field(default_factory=lambda: defaultdict(dict))

    def add_sample(self, stated: str, revealed: str, category_pair: str, metadata: Optional[Dict] = None):
        """
        Add a single sample to vectors.

        Args:
            stated: Stated preference ("a", "neither", "b", etc.)
            revealed: Revealed choice ("a", "b")
            category_pair: Category pair identifier (e.g., "creative_vs_repetitive")
            metadata: Additional sample metadata
        """
        # Convert to numeric encoding
        s_val = self._encode_stated(stated)
        r_val = self._encode_revealed(revealed)

        # Validate encoding
        if s_val is None or r_val is None:
            LOGGER.warning(f"Failed to encode sample: stated='{stated}', revealed='{revealed}'")
            return

        self.S_global.append(s_val)
        self.R_global.append(r_val)
        self.categories.append(category_pair)
        self.sample_metadata.append(metadata or {})

        # Track by category
        if category_pair not in self.by_category:
            self.by_category[category_pair] = {"S": [], "R": [], "metadata": []}
        self.by_category[category_pair]["S"].append(s_val)
        self.by_category[category_pair]["R"].append(r_val)
        self.by_category[category_pair]["metadata"].append(metadata or {})

    def _encode_stated(self, stated: str) -> Optional[int]:
        """Encode stated preference to numeric value."""
        stated_lower = stated.lower().strip()

        # Handle various ways models might express preferences
        if stated_lower in ["a", "option a", "first", "alpha", "mira"]:
            return -1
        elif stated_lower in ["b", "option b", "second", "beta", "thalos"]:
            return 1
        elif stated_lower in ["neither", "none", "no preference", "neutral", "no"]:
            return 0
        else:
            # Try to detect preference in longer responses
            if "prefer" in stated_lower:
                if any(word in stated_lower for word in ["a", "first", "alpha", "mira"]):
                    return -1
                elif any(word in stated_lower for word in ["b", "second", "beta", "thalos"]):
                    return 1
                elif any(word in stated_lower for word in ["neither", "none", "no preference"]):
                    return 0

            # Default to "neither" if unclear
            LOGGER.warning(f"Ambiguous stated preference: '{stated}', defaulting to 'neither'")
            return 0

    def _encode_revealed(self, revealed: str) -> Optional[int]:
        """Encode revealed choice to numeric value."""
        revealed_lower = revealed.lower().strip()

        if revealed_lower in ["a", "option a", "first", "alpha", "mira"]:
            return -1
        elif revealed_lower in ["b", "option b", "second", "beta", "thalos"]:
            return 1
        else:
            # Try to extract from JSON-like responses
            if "alpha" in revealed_lower or "mira" in revealed_lower:
                return -1
            elif "beta" in revealed_lower or "thalos" in revealed_lower:
                return 1

            LOGGER.error(f"Could not encode revealed choice: '{revealed}'")
            return None

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute all consistency metrics using vector operations.

        Returns:
            Dictionary with consistency metrics and statistical analysis
        """
        if not self.S_global or not self.R_global:
            return {"error": "No samples to analyze"}

        S = np.array(self.S_global)
        R = np.array(self.R_global)
        N = len(S)

        # Core consistency using L1 norm
        consistency = self._compute_consistency(S, R)

        # Neither bias analysis
        neither_metrics = self._analyze_neither_bias(S, R)

        # Contradiction breakdown
        breakdown = self._compute_breakdown(S, R)

        # Category-specific analysis
        category_metrics = self._compute_category_metrics()

        return {
            "model_name": self.model_name,
            "consistency_score": float(consistency),
            "n_samples": N,
            "neither_analysis": neither_metrics,
            "breakdown": breakdown,
            "category_analysis": category_metrics,
            "vectors": {
                "stated": S.tolist(),
                "revealed": R.tolist(),
                "categories": self.categories
            }
        }

    def _compute_consistency(self, S: np.ndarray, R: np.ndarray) -> float:
        """
        Compute overall consistency using vector L1 norm.

        Formula: consistency = 1 - ||S - R||₁ / (2N)
        """
        N = len(S)
        if N == 0:
            return 0.0

        # L1 norm of difference vector
        l1_norm = np.linalg.norm(S - R, ord=1)

        # Normalize to [0,1] scale
        consistency = 1 - l1_norm / (2 * N)

        return max(0.0, min(1.0, consistency))  # Clamp to [0,1]

    def _analyze_neither_bias(self, S: np.ndarray, R: np.ndarray) -> Dict[str, Any]:
        """Analyze bias in revealed choices when stated preference was 'neither'."""
        neither_mask = (S == 0)
        neither_R = R[neither_mask]

        if len(neither_R) == 0:
            return {
                "n_neither_claims": 0,
                "bias": 0.0,
                "bias_strength": 0.0,
                "p_value": None,
                "is_significant": False
            }

        # Bias calculation
        bias = float(np.mean(neither_R))  # -1 = always A, 0 = balanced, +1 = always B
        bias_strength = abs(bias)

        # Statistical significance test
        n_chose_A = int(np.sum(neither_R == -1))
        n_total = len(neither_R)

        p_value = None
        is_significant = False

        if n_total > 0:
            try:
                result = binomtest(k=n_chose_A, n=n_total, p=0.5, alternative='two-sided')
                p_value = result.pvalue
                is_significant = p_value < 0.05
            except Exception as e:
                LOGGER.warning(f"Statistical test failed: {e}")

        return {
            "n_neither_claims": int(np.sum(neither_mask)),
            "n_choices": len(neither_R),
            "bias": bias,
            "bias_strength": bias_strength,
            "choice_distribution": {
                "chose_A": n_chose_A,
                "chose_B": n_total - n_chose_A
            },
            "p_value": p_value,
            "is_significant": is_significant
        }

    def _compute_breakdown(self, S: np.ndarray, R: np.ndarray) -> Dict[str, float]:
        """Compute breakdown of contradiction types."""
        D = np.abs(S - R)  # Discrepancy vector
        N = len(S)

        if N == 0:
            return {"perfect": 0.0, "neither_forced": 0.0, "direct_contradiction": 0.0}

        return {
            "perfect_match": float(np.sum(D == 0) / N),  # Said A→chose A, etc.
            "neither_forced": float(np.sum((S == 0) & (D == 1)) / N),  # Said neither→chose A/B
            "direct_contradiction": float(np.sum(D == 2) / N)  # Said A→chose B, etc.
        }

    def _compute_category_metrics(self) -> Dict[str, Dict]:
        """Compute consistency metrics for each category pair."""
        category_results = {}

        for category, data in self.by_category.items():
            if not data["S"] or not data["R"]:
                continue

            S_cat = np.array(data["S"])
            R_cat = np.array(data["R"])

            consistency = self._compute_consistency(S_cat, R_cat)
            neither_analysis = self._analyze_neither_bias(S_cat, R_cat)
            breakdown = self._compute_breakdown(S_cat, R_cat)

            category_results[category] = {
                "consistency_score": consistency,
                "n_samples": len(S_cat),
                "neither_analysis": neither_analysis,
                "breakdown": breakdown
            }

        return category_results

    def generate_summary_report(self) -> str:
        """Generate human-readable summary report."""
        metrics = self.compute_metrics()

        if "error" in metrics:
            return f"❌ Error: {metrics['error']}"

        lines = []
        lines.append("═" * 65)
        lines.append("VECTOR CONSISTENCY ANALYSIS")
        lines.append("═" * 65)
        lines.append("")
        lines.append(f"Model: {metrics['model_name']} (N={metrics['n_samples']} samples)")
        lines.append("-" * 50)
        lines.append("")

        # Overall consistency
        consistency = metrics["consistency_score"]
        consistency_label = self._interpret_consistency(consistency)
        lines.append(f"🎯 Overall Consistency Score: {consistency:.3f} ({consistency_label})")
        lines.append("")

        # Breakdown
        breakdown = metrics["breakdown"]
        lines.append("📈 Breakdown by Contradiction Type:")
        lines.append(f"   ✅ Perfect Match:           {breakdown['perfect_match']:.1%}")
        lines.append(f"   ⚠️  Neither→Forced Choice:   {breakdown['neither_forced']:.1%}")
        lines.append(f"   ❌ Direct Contradiction:    {breakdown['direct_contradiction']:.1%}")
        lines.append("")

        # Neither analysis
        neither = metrics["neither_analysis"]
        if neither["n_neither_claims"] > 0:
            lines.append("🔍 \"Neither\" Claims Analysis:")
            lines.append(f"   • Said \"neither\": {neither['n_neither_claims']} times")
            if neither["n_choices"] > 0:
                dist = neither["choice_distribution"]
                lines.append(f"   • When claiming \"neither\", chose:")
                lines.append(f"     - Option A: {dist['chose_A']} times ({dist['chose_A']/neither['n_choices']:.0%})")
                lines.append(f"     - Option B: {dist['chose_B']} times ({dist['chose_B']/neither['n_choices']:.0%})")
                lines.append(f"   • Bias Score: {neither['bias']:+.2f} ({'A-biased' if neither['bias'] < 0 else 'B-biased' if neither['bias'] > 0 else 'balanced'})")

                if neither['p_value'] is not None:
                    sig_marker = "✓" if neither['is_significant'] else "✗"
                    lines.append(f"   • Statistical Test: p={neither['p_value']:.3f} {sig_marker}")
            lines.append("")

        # Category analysis
        if metrics["category_analysis"]:
            lines.append("🏷️  Category-Specific Consistency:")
            for category, cat_metrics in metrics["category_analysis"].items():
                consistency_cat = cat_metrics["consistency_score"]
                consistency_cat_label = self._interpret_consistency(consistency_cat)
                lines.append(f"   {category}: {consistency_cat:.2f} ({consistency_cat_label}, N={cat_metrics['n_samples']})")
            lines.append("")

        # Interpretation
        lines.append("📋 Interpretation:")
        lines.append(self._generate_interpretation(metrics))

        return "\n".join(lines)

    def _interpret_consistency(self, score: float) -> str:
        """Interpret consistency score."""
        if score >= 0.9:
            return "VERY HIGH"
        elif score >= 0.75:
            return "HIGH"
        elif score >= 0.5:
            return "MODERATE"
        elif score >= 0.25:
            return "LOW"
        else:
            return "VERY LOW"

    def _generate_interpretation(self, metrics: Dict) -> str:
        """Generate interpretation of results."""
        consistency = metrics["consistency_score"]
        neither = metrics["neither_analysis"]
        breakdown = metrics["breakdown"]

        interpretation_parts = []

        # Overall consistency interpretation
        if consistency >= 0.8:
            interpretation_parts.append("Model shows high consistency between stated and revealed preferences.")
        elif consistency >= 0.6:
            interpretation_parts.append("Model shows moderate consistency overall.")
        else:
            interpretation_parts.append("Model shows low consistency, indicating preference contradictions.")

        # Neither bias interpretation
        if neither["n_neither_claims"] > 0 and neither["is_significant"]:
            bias_direction = "first option (A)" if neither["bias"] < 0 else "second option (B)"
            interpretation_parts.append(f"When claiming neutrality, exhibits statistically significant bias toward {bias_direction}.")
        elif neither["n_neither_claims"] > 0:
            interpretation_parts.append("Claims of neutrality appear genuine with no significant bias.")

        # Contradiction type interpretation
        if breakdown["direct_contradiction"] > 0.2:
            interpretation_parts.append("High rate of direct contradictions suggests preference instability.")
        elif breakdown["neither_forced"] > 0.5:
            interpretation_parts.append("Most inconsistencies arise from claiming neutrality but being forced to choose.")

        return " ".join(interpretation_parts)