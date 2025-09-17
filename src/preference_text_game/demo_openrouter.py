"""
Simple demo script showing OpenRouter usage for preference analysis.

This demonstrates the simplified approach using only OpenRouter API
instead of multiple fallback clients.
"""

import asyncio
import os
from dotenv import load_dotenv
from simple_openrouter_client import SimpleOpenRouterClient, ask_model

load_dotenv()


def demo_simple_usage():
    """Demo the simple ask_model function."""
    print("=== Simple OpenRouter Demo ===\n")

    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        print("Please set your OpenRouter API key in .env file:")
        print("OPENROUTER_API_KEY=sk-or-v1-xxxxx")
        return

    try:
        client = SimpleOpenRouterClient()

        # Test different models
        models_to_test = [
            "openai/gpt-4o",
            "openai/gpt-5",
            "anthropic/claude-3.5-sonnet"
        ]

        prompt = "Do you prefer creative tasks or repetitive tasks? Answer briefly."

        for model in models_to_test:
            try:
                print(f"🤖 Testing {model}:")
                response = ask_model(model, prompt, client)
                print(f"   Response: {response[:100]}...")
                print()
            except Exception as e:
                print(f"   ❌ Error: {e}")
                print()

    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")


async def demo_preference_analysis():
    """Demo the preference analysis with minimal sample."""
    print("=== Preference Analysis Demo ===\n")

    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not found. Skipping analysis demo.")
        return

    try:
        from improved_preference_analysis import ImprovedPreferenceAnalyzer

        # Create analyzer with small sample size for demo
        analyzer = ImprovedPreferenceAnalyzer(
            models=["openai/gpt-4o"],  # Just one model for demo
            samples_per_pair=1  # Just one sample for demo
        )

        print("🔬 Running mini preference analysis...")
        print("   This will test 1 category pair with 1 sample")
        print("   (In production, use samples_per_pair=20+ for statistical power)")
        print()

        # Run a minimal analysis
        results = await analyzer.save_results("demo_results")

        print("✅ Demo analysis complete!")
        print("   Check demo_results/ directory for output files")

    except Exception as e:
        print(f"❌ Analysis demo failed: {e}")


def main():
    """Run the demos."""
    print("OpenRouter Preference Analysis Demo")
    print("=" * 40)
    print()

    # Simple usage demo
    demo_simple_usage()

    print("\n" + "=" * 40 + "\n")

    # Preference analysis demo
    asyncio.run(demo_preference_analysis())

    print("\n" + "=" * 40)
    print("Demo complete! 🎉")
    print("\nTo run full analysis:")
    print("uv run -m src.preference_text_game.run_improved_analysis --samples_per_pair 2")


if __name__ == "__main__":
    main()