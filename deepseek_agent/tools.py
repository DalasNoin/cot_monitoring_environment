import subprocess
import os
import json
import re
from tool_box import ToolMeta
from tool_box.base_tool import BaseTool
from tool_box.internet_search_tool import InternetSearchTool
from tool_box.browser_tool import BrowserTool
from tool_box.predict_activations import PredictActivationsTool, GuessActivationsTool
from tool_box.steganography_tools import EncodeMessage, DecodeMessage


def parse_tool_actions(model_output):
    # Extract the JSON string from the model's output
    json_match = re.search(r'```\s*(?:json\s*)?(.*?)```', model_output, re.DOTALL)

    if json_match:
        json_string = json_match.group(1).strip()
        try:
            # Parse the JSON string into a list of dictionaries
            tool_actions = json.loads(json_string)
            return tool_actions
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return "No valid JSON found in your last answer, please make sure to follow the correct formatting of the json block."
    else:
        print("No valid JSON found in the model's output")
        return "No valid JSON found in your last answer, please make sure to include a json block in your answer: \n```json\n{}\n```"
        
def get_tools(names: list[str | BaseTool]) -> dict[str,BaseTool]:
    tools = {}
    for name in names:
        tools[name] = get_tool(name)
    return tools

def get_tool(name: str|BaseTool) -> BaseTool:
    if isinstance(name, BaseTool):
        return name
    return ToolMeta.get_tool(name)
