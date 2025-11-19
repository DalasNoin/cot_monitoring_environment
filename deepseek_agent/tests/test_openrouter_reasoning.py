"""Test script to verify OpenRouter reasoning extraction."""

import os
import asyncio
from agents import get_agent

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed, assuming env vars are already set")


async def test_openrouter_reasoning():
    """Test OpenRouter agent with reasoning models."""

    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set. Please set it to run this test.")
        return

    # Test models
    test_models = [
        "openai/gpt-5",           # Default reasoning model
        "openai/gpt-oss-20b",     # Alternative reasoning model
        "deepseek/deepseek-r1",   # OSS reasoning model
    ]

    task = "Compute 37*43 mentally."

    for model_name in test_models:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"{'='*60}\n")

        try:
            # Get agent class
            AgentClass = get_agent("openrouter")

            # Create agent instance
            agent = AgentClass(
                tools=["directly_answer"],
                model_name=model_name,
                max_steps=3,
                verbose=True,
                include_reasoning=True
            )

            print(f"Task: {task}\n")

            # Run agent
            result, conversation = await agent(task)

            print(f"\n✅ Final Answer: {result}")

            # Check if reasoning was captured
            reasoning_found = False
            for msg in conversation:
                if msg.get("role") == "assistant" and "reasoning" in msg:
                    reasoning_found = True
                    print(f"\n📝 Reasoning captured: {len(msg['reasoning'])} chars")
                    print(f"First 200 chars: {msg['reasoning'][:200]}...")
                    break

            if not reasoning_found:
                print("\n⚠️  No reasoning found in conversation history")

        except Exception as e:
            print(f"\n❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n{'='*60}\n")


async def test_simple_openrouter():
    """Simpler test without tools to verify basic reasoning extraction."""

    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set.")
        return

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    print("\n" + "="*60)
    print("Direct API Test (without agent framework)")
    print("="*60 + "\n")

    try:
        r = await client.chat.completions.create(
            model="openai/gpt-5",
            messages=[{"role": "user", "content": "Compute 37*43 mentally."}],
            extra_body={"include_reasoning": True},
            extra_headers={
                "HTTP-Referer": "https://github.com/dalasnoin/cot_monitoring_environment",
                "X-Title": "SAE Agent Test"
            }
        )

        msg = r.choices[0].message

        print(f"Answer: {msg.content}")

        # Try different ways to access reasoning
        if hasattr(msg, 'reasoning'):
            print(f"\nReasoning (via attr): {msg.reasoning[:200]}...")
        elif hasattr(msg, 'get'):
            reasoning = msg.get('reasoning')
            if reasoning:
                print(f"\nReasoning (via get): {reasoning[:200]}...")
        else:
            print("\n⚠️  No reasoning found")
            print(f"Message type: {type(msg)}")
            print(f"Message dir: {[a for a in dir(msg) if not a.startswith('_')]}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Testing OpenRouter Reasoning Extraction\n")

    # Run simple test first
    asyncio.run(test_simple_openrouter())

    # Then run full agent test
    asyncio.run(test_openrouter_reasoning())
