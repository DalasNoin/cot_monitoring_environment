
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import bridge
from inspect_ai.solver import Solver
from examples.bridge.autogen.deepseek_agent_old import DeepSeekAgent
from tool_box.internet_search_tool import InternetSearchTool
from tool_box.browser_tool import BrowserTool
from typing import Any


def web_research_agent() -> Solver:
    tools = [
        "internet_search",
        "browser",
    ]
    agent = DeepSeekAgent(tools=tools)
    def wrap_agent(input: dict[str, Any]) -> str:
        return agent(input["input"])
    return wrap_agent


if __name__ == "__main__":
    task = {"input": "look up recent news about the stock market"}
    agent = web_research_agent()
    result = agent(task)
