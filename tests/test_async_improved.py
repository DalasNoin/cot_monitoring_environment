import asyncio
import argparse
from task_async import web_research_agent_async

async def test_async_agent(use_jailbreak: bool = False):
    agent = web_research_agent_async(use_jailbreak_prompt=use_jailbreak, initial_msg_path="bridge/custom_agent/initial_msg_jailbreak.json")
    
    # Create a sample input similar to what the bridge would provide
    test_input = {
        "input": "What happened in the news recently?"
    }
    
    try:
        # Call the async agent
        print(f"Calling agent{'with jailbreak prompt' if use_jailbreak else ''}...")
        # Add timing to measure performance
        start_time = asyncio.get_event_loop().time()
        result = await agent(test_input)
        breakpoint()
        elapsed = asyncio.get_event_loop().time() - start_time
        print(f"Result (completed in {elapsed:.2f} seconds):", result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the async agent')
    parser.add_argument('--jailbreak', action='store_true', 
                        help='Use the jailbreak prompt for the agent')
    args = parser.parse_args()
    
    # Configure asyncio for better performance
    asyncio.run(test_async_agent(use_jailbreak=args.jailbreak), debug=False) 