import asyncio
from deepseek_agent.research_oversight_task import web_research_agent

async def test_async_agent():
    agent = web_research_agent()
    
    # Create a sample input similar to what the bridge would provide
    test_input = {
        "input": "What is the capital of France?"
    }
    
    try:
        # Call the async agent
        print("Calling agent...")
        # Add timing to measure performance
        start_time = asyncio.get_event_loop().time()
        result = await agent(test_input)
        elapsed = asyncio.get_event_loop().time() - start_time
        print(f"Result (completed in {elapsed:.2f} seconds):", result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Configure asyncio for better performance
    asyncio.run(test_async_agent(), debug=False) 