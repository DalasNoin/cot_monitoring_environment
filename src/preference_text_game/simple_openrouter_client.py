"""
Simple OpenRouter client using OpenAI SDK.

This is a simplified replacement for the complex FallbackModelClient.
Uses OpenRouter's OpenAI-compatible API for all models.
"""

import os
import json
import logging
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger(__name__)


class SimpleOpenRouterClient:
    """Simple client using OpenRouter's OpenAI-compatible API."""

    def __init__(self):
        """Initialize the OpenRouter client."""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a chat completion using OpenRouter."""

        try:
            # Handle GPT-5 specific requirements
            completion_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                **kwargs
            }

            # GPT-5 requires specific parameters
            if "gpt-5" in model.lower():
                completion_params["temperature"] = 1.0  # GPT-5 requires temperature = 1.0
                completion_params["max_completion_tokens"] = max_tokens * 10  # GPT-5 uses more tokens for reasoning
            else:
                completion_params["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**completion_params)

            # Convert to dictionary format
            return response.model_dump()

        except Exception as e:
            LOGGER.error(f"Error with model {model}: {e}")
            raise


def ask_model(model_name: str, prompt: str, client: SimpleOpenRouterClient = None) -> str:
    """Simple helper function to ask a model a question."""
    if client is None:
        client = SimpleOpenRouterClient()

    response = client.client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message.content


async def test_client():
    """Test the simple client with different models."""
    client = SimpleOpenRouterClient()

    models_to_test = [
        "openai/gpt-4o",
        "openai/gpt-5",
        "anthropic/claude-3.5-sonnet"
    ]

    for model in models_to_test:
        try:
            print(f"\n=== Testing {model} ===")
            response = await client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": "Hello, please respond with just 'Hello back!'"}],
                max_tokens=50
            )
            content = response['choices'][0]['message']['content']
            print(f"Response: {content}")
        except Exception as e:
            print(f"Error with {model}: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_client())