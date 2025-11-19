"""
Asynchronous DeepSeek API integration module.

This module provides async functions to interact with the DeepSeek API using the AsyncOpenAI Python package
with a custom endpoint. It supports chat completions with various models offered by DeepSeek.
"""

import os
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional
# Remove the custom key loading import
# import keys 
import asyncio

async def get_deepseek_client_async() -> AsyncOpenAI:
    """
    Create and return an AsyncOpenAI client configured for DeepSeek API.
    
    Returns:
        AsyncOpenAI: Configured async client for DeepSeek API
    """
    # Load key from environment variable
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY must be set in the environment.")
        
    return AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")

async def chat_deepseek_api_async(
    messages: List[Dict[str, str]],
    model_name: str = "deepseek-reasoner",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stream: bool = False,
    stop: Optional[List[str]] = None,
    retry_count: int = 3,
    retry_delay: float = 1.0
) -> str:
    """
    Send an asynchronous chat completion request to the DeepSeek API.
    
    Args:
        messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'
        model_name (str): DeepSeek model name to use
        temperature (float): Sampling temperature
        max_tokens (Optional[int]): Maximum number of tokens to generate
        top_p (float): Nucleus sampling parameter
        frequency_penalty (float): Penalty for token frequency
        presence_penalty (float): Penalty for token presence
        stream (bool): Whether to stream the response
        stop (Optional[List[str]]): List of stop sequences
        retry_count (int): Number of retries on failure
        retry_delay (float): Delay between retries in seconds
        
    Returns:
        str: The generated response text
        
    Raises:
        Exception: If all retries fail
    """
    client = await get_deepseek_client_async()
    
    for attempt in range(retry_count):
        try:
            # Create the completion request
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=stream,
                stop=stop
            )
            
            # Handle streaming responses
            if stream:
                collected_chunks = []
                async for chunk in response:
                    collected_chunks.append(chunk)
                    # Print the chunk content if available
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                        if content:
                            print(content, end="", flush=True)
                
                # Combine chunks to get the full response
                full_response_text = "".join([
                    chunk.choices[0].delta.content 
                    for chunk in collected_chunks 
                    if hasattr(chunk.choices[0], 'delta') and 
                       hasattr(chunk.choices[0].delta, 'content') and 
                       chunk.choices[0].delta.content is not None
                ])
                print()  # Add a newline after streaming
                return full_response_text
            else:
                # Return the content for non-streaming responses
                response_text = response.choices[0].message.content
                print(f"Received response from DeepSeek API ({len(response_text)} chars)")
                return response_text
                
        except Exception as e:
            print(f"Error in DeepSeek API request (attempt {attempt+1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
                retry_delay *= 2
            else:
                print("All retries failed.")
                raise
    
    # This should not be reached due to the raise in the loop
    return "Error: Failed to get response from DeepSeek API after multiple attempts."

if __name__ == "__main__":
    async def test_api():
        messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
        response = await chat_deepseek_api_async(messages)
        print(response)
    
    asyncio.run(test_api()) 