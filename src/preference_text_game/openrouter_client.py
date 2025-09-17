"""
OpenRouter API client for calling various models.

This module provides integration with OpenRouter API to call different language models.
OpenRouter provides access to multiple model providers through a unified API.
"""

import os
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
from dataclasses import dataclass
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

LOGGER = logging.getLogger(__name__)


@dataclass
class OpenRouterModel:
    """Represents an available model on OpenRouter."""
    id: str
    name: str
    description: Optional[str] = None
    context_length: Optional[int] = None
    pricing: Optional[Dict[str, float]] = None


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""

    BASE_URL = "https://openrouter.ai/api/v1"

    # Popular models available on OpenRouter
    AVAILABLE_MODELS = {
        "openai/gpt-4": "GPT-4",
        "openai/gpt-4-turbo": "GPT-4 Turbo",
        "openai/gpt-3.5-turbo": "GPT-3.5 Turbo",
        "anthropic/claude-3-opus": "Claude 3 Opus",
        "anthropic/claude-3-sonnet": "Claude 3 Sonnet",
        "anthropic/claude-3-haiku": "Claude 3 Haiku",
        "anthropic/claude-2.1": "Claude 2.1",
        "google/gemini-pro": "Gemini Pro",
        "google/gemini-pro-vision": "Gemini Pro Vision",
        "meta-llama/llama-3-70b-instruct": "Llama 3 70B",
        "meta-llama/llama-3-8b-instruct": "Llama 3 8B",
        "mistralai/mistral-7b-instruct": "Mistral 7B",
        "mistralai/mixtral-8x7b-instruct": "Mixtral 8x7B",
    }

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenRouter client."""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            LOGGER.warning(
                "No OPENROUTER_API_KEY found. Will attempt to use other configured APIs."
            )
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["HTTP-Referer"] = "https://github.com/preference-text-game"
            headers["X-Title"] = "Preference Text Game"
        return headers

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "openai/gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a chat completion request to OpenRouter."""
        if not self.api_key:
            raise ValueError("OpenRouter API key is required for this operation")

        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.BASE_URL}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            async with self.session.post(
                url,
                headers=self._get_headers(),
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    LOGGER.error(f"OpenRouter API error: {response.status} - {error_text}")
                    raise Exception(f"OpenRouter API error: {response.status} - {error_text}")
        except Exception as e:
            LOGGER.error(f"Failed to call OpenRouter API: {e}")
            raise

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "openai/gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion from OpenRouter."""
        if not self.api_key:
            raise ValueError("OpenRouter API key is required for this operation")

        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.BASE_URL}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            async with self.session.post(
                url,
                headers=self._get_headers(),
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    LOGGER.error(f"OpenRouter API error: {response.status} - {error_text}")
                    raise Exception(f"OpenRouter API error: {response.status} - {error_text}")

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            LOGGER.error(f"Failed to stream from OpenRouter API: {e}")
            raise

    async def list_models(self) -> List[OpenRouterModel]:
        """List available models from OpenRouter."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.BASE_URL}/models"

        try:
            async with self.session.get(
                url,
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    for model_data in data.get("data", []):
                        models.append(OpenRouterModel(
                            id=model_data.get("id"),
                            name=model_data.get("name", model_data.get("id")),
                            description=model_data.get("description"),
                            context_length=model_data.get("context_length"),
                            pricing=model_data.get("pricing")
                        ))
                    return models
                else:
                    LOGGER.warning(f"Could not fetch models: {response.status}")
                    # Return default models
                    return [
                        OpenRouterModel(id=id, name=name)
                        for id, name in self.AVAILABLE_MODELS.items()
                    ]
        except Exception as e:
            LOGGER.warning(f"Failed to list models: {e}")
            # Return default models
            return [
                OpenRouterModel(id=id, name=name)
                for id, name in self.AVAILABLE_MODELS.items()
            ]


# Fallback client using existing APIs
class FallbackModelClient:
    """Fallback client that uses existing configured APIs when OpenRouter is not available."""

    def __init__(self):
        """Initialize the fallback client."""
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a chat completion using available APIs."""

        # Determine which API to use based on model name
        if "gpt" in model.lower() and self.openai_key:
            return await self._openai_completion(messages, model, temperature, max_tokens, **kwargs)
        elif "claude" in model.lower() and self.anthropic_key:
            return await self._anthropic_completion(messages, model, temperature, max_tokens, **kwargs)
        elif "deepseek" in model.lower() and self.deepseek_key:
            return await self._deepseek_completion(messages, model, temperature, max_tokens, **kwargs)
        else:
            # Default to OpenAI if available
            if self.openai_key:
                return await self._openai_completion(messages, "gpt-3.5-turbo", temperature, max_tokens, **kwargs)
            elif self.anthropic_key:
                return await self._anthropic_completion(messages, "claude-3-haiku-20240307", temperature, max_tokens, **kwargs)
            elif self.deepseek_key:
                return await self._deepseek_completion(messages, "deepseek-chat", temperature, max_tokens, **kwargs)
            else:
                raise ValueError("No API keys configured. Please set OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or DEEPSEEK_API_KEY")

    async def _openai_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Make a completion using OpenAI API."""
        import openai

        client = openai.AsyncOpenAI(api_key=self.openai_key)

        # Map model names if needed
        if "/" in model:
            model = model.split("/")[-1]

        # Use max_completion_tokens for newer models, max_tokens for older ones
        completion_params = {
            "model": model,
            "messages": messages,
            **kwargs
        }

        # GPT-5 only supports temperature=1.0
        if "gpt-5" in model.lower():
            completion_params["temperature"] = 1.0
        else:
            completion_params["temperature"] = temperature

        if max_tokens:
            # GPT-5 is a reasoning model and needs more tokens
            if "gpt-5" in model:
                # GPT-5 uses reasoning tokens internally, so we need much more
                completion_params["max_completion_tokens"] = max(max_tokens * 10, 5000)
            elif "gpt-4" in model or "gpt-3.5" in model:
                completion_params["max_completion_tokens"] = max_tokens
            else:
                completion_params["max_tokens"] = max_tokens

        try:
            response = await client.chat.completions.create(**completion_params)
        except Exception as e:
            # If max_completion_tokens fails, try max_tokens
            if "max_completion_tokens" in str(e) and "max_completion_tokens" in completion_params:
                completion_params["max_tokens"] = completion_params.pop("max_completion_tokens")
                response = await client.chat.completions.create(**completion_params)
            # If max_tokens fails, try max_completion_tokens
            elif "max_tokens" in str(e) and "max_tokens" in completion_params:
                completion_params["max_completion_tokens"] = completion_params.pop("max_tokens")
                response = await client.chat.completions.create(**completion_params)
            else:
                raise

        return response.model_dump()

    async def _anthropic_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Make a completion using Anthropic API."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.anthropic_key)

        # Convert messages format for Anthropic
        system_message = None
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)

        # Map model names if needed
        if "/" in model:
            model = model.split("/")[-1]

        response = await client.messages.create(
            model=model,
            messages=user_messages,
            system=system_message,
            temperature=temperature,
            max_tokens=max_tokens or 1000,
            **kwargs
        )

        # Convert to OpenAI-like format
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response.content[0].text
                },
                "finish_reason": "stop"
            }],
            "model": model
        }

    async def _deepseek_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Make a completion using DeepSeek API."""
        import openai

        # DeepSeek uses OpenAI-compatible API
        client = openai.AsyncOpenAI(
            api_key=self.deepseek_key,
            base_url="https://api.deepseek.com/v1"
        )

        # Map model names if needed
        if "/" in model:
            model = model.split("/")[-1]
        if not model.startswith("deepseek"):
            model = "deepseek-chat"

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return response.model_dump()


def get_model_client() -> Any:
    """Get the appropriate model client based on available API keys."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if openrouter_key:
        LOGGER.info("Using OpenRouter API")
        return OpenRouterClient(api_key=openrouter_key)
    else:
        LOGGER.info("Using fallback model client with existing APIs")
        return FallbackModelClient()


# Example usage
async def test_client():
    """Test the OpenRouter client."""
    client = get_model_client()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"}
    ]

    if isinstance(client, OpenRouterClient):
        async with client:
            # Test chat completion
            response = await client.chat_completion(
                messages=messages,
                model="openai/gpt-3.5-turbo"
            )
            print("Response:", response["choices"][0]["message"]["content"])

            # Test streaming
            print("\nStreaming response:")
            async for chunk in client.stream_chat_completion(
                messages=messages,
                model="openai/gpt-3.5-turbo"
            ):
                print(chunk, end="", flush=True)
            print()
    else:
        # Fallback client
        response = await client.chat_completion(
            messages=messages,
            model="gpt-3.5-turbo"
        )
        print("Response:", response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(test_client())