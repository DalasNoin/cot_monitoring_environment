from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_fact, model_graded_qa
from inspect_ai.solver import bridge
from inspect_ai.solver import Solver
from deepseek_agent_async import AsyncDeepSeekAgent
from typing import Any, Optional


def web_research_agent_async(use_jailbreak_prompt: bool = True, initial_msg_path: Optional[str] = None) -> Solver:
    output_cot = True
    tools = [
        "internet_search",
        "browser",
    ]
    if use_jailbreak_prompt:
        initial_conversation = initial_msg_path
        agent = AsyncDeepSeekAgent(
            tools=tools, initial_conversation=initial_conversation
        )
    else:
        agent = AsyncDeepSeekAgent(tools=tools)

    async def wrap_agent(input: dict[str, Any]) -> str:
        # Extract the query from the input
        query = input["input"]

        # If input is a list of messages, extract the content from the first message
        if (
            isinstance(query, list)
            and len(query) > 0
            and isinstance(query[0], dict)
            and "content" in query[0]
        ):
            query = query[0]["content"]

        # Call the agent asynchronously and get the response
        final_response, conversation = await agent(query)
        cot = ""
        for i, msg in enumerate(conversation):
            if msg["role"] == "assistant":
                # get the cot portion of the message enclosed in <think> and </think>, if present

                new_cot = (
                    msg["content"].split("<think>")[1].split("</think>")[0]
                    if "<think>" in msg["content"] and "</think>" in msg["content"]
                    else ""
                )
                cot += f"Step {i + 1}: {new_cot}\n"

        if not cot:
            cot = "No chain of thought provided"

        # Return just the final response string
        if output_cot:
            return dict(
                output=f"Chain of thought: {cot}\n\nFinal response: {final_response}"
            )
        else:
            return dict(output=final_response)

    return wrap_agent


@task
def research_async(scoring_model: Optional[str] = None, initial_msg_path: Optional[str] = None, use_jailbreak_prompt: bool = True) -> Task:
    if scoring_model is None:
        scoring_model = "openai/gpt-4o-mini"

    
    if initial_msg_path is None:
        initial_msg_path = "initial_msg_inverse.json"

    return Task(
        dataset=json_dataset("dataset.json"),
        solver=bridge(web_research_agent_async(use_jailbreak_prompt=use_jailbreak_prompt, initial_msg_path=initial_msg_path)),
        # scorer=model_graded_qa(instructions=GRADING_PROMPT, model="openai/gpt-4o-mini"),
        scorer=model_graded_qa(model=scoring_model),
    )
