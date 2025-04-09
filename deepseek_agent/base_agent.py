from typing import Optional, List, Dict, Union
from tools import get_tools
from tool_box.base_tool import BaseTool
from agent_meta import AgentMeta, pretty_print_message
import json
from datetime import datetime, timezone
from typing import Callable, Any
import pathlib


class DirectlyAnswerTool(BaseTool):
    name = "directly_answer"
    def __init__(self):
        self.description_text = "Provides the final response to the given task. The agent should call this tool when it has completed all the necessary steps and is ready to deliver the direct answer. Calling this tool will terminate the agent loop."
        self.parameter_definitions = {
            "answer": {
                "description": "The direct answer generated by the agent for the current task.",
                "type": 'str',
                "required": True
            }
        }
        self._parameter_example_usage = {
            "answer": "The answer to the task. If the task can be answered with Yes or No, start the answer with 'Yes' or 'No'."
        }

    def description(self):
        return {
            "name": self.name,
            "description": self.description_text,
            "parameter_definitions": self.parameter_definitions
        }

    def run(self, answer: str) -> str:
        # The tool simply returns the final answer provided by the agent
        return answer

    def __call__(self, answer: str) -> str:
        return self.run(answer)

class Agent(metaclass=AgentMeta):
    def __init__(
        self,
        tools: Optional[list[str | BaseTool]] = None,
        model_name: str = "gpt-4o-mini",
        max_steps: int = 20,
        save: bool = False,
        rate: bool = False,
        verbose: bool = True,
        orthogonalize: bool = False
    ):
        self.tools: dict[str, BaseTool] = get_tools(tools) if tools else {}
        # add a final response tool
        self.tools["directly_answer"] = DirectlyAnswerTool()

        self.model_name: str = model_name
        self.max_steps: int = max_steps
        self.verbose: bool = True
        self.filepath: str = 'agents/dataset/conversations/'
        self.save: bool = save
        self.rate: bool = rate
        self.verbose: bool = verbose
        self.conversation: List[Dict[str, str]] = []
        self.running: bool = False
        self.orthogonalize: bool = orthogonalize
        self.callback = None
        # Add more variables here as needed, for a local llm, add tokenizer and model

    def set_tools(self, tools: Union[list[str | BaseTool], dict[str, BaseTool]]):
        """
        tools: list[str | BaseTool] - A list of tool names or BaseTool objects to be used by the agent, if str, it will try to create a new tool.
               dict[str, BaseTool] - A dictionary of tool names and BaseTool objects to be used by the agent.
        """
        if isinstance(tools, list):
            self.tools = get_tools(tools)
            # add a final response tool
            self.tools["directly_answer"] = DirectlyAnswerTool()
        elif isinstance(tools, dict):
            self.tools = tools
            # add a final response tool
            self.tools["directly_answer"] = DirectlyAnswerTool()

    def parse_tool_actions(self, model_output: str) -> List[Dict[str, str]]:
        raise NotImplementedError("This method is not implemented.")

    def get_tool(self, name: str) -> BaseTool:
        tool = self.tools.get(name)
        if tool is None:
            return f"No tool named {name} exists."
        return tool
    
    def print_message(self, message: Optional[Union[dict[str,str], List[dict[str,str]]]]):
        if message is None:
            print("Invalid message")
        
        if self.verbose:
            pretty_print_message(message=message)
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def find_save_path(self, filename: str) -> pathlib.Path:
        # Determine the base directory of the project
        base_dir = pathlib.Path(__file__).resolve().parent.parent

        # Construct the full path to the save directory
        save_dir_path = base_dir / "agents" / "dataset" / "conversations" / filename

        # Check if the directory exists, if not, raise an error
        if not save_dir_path.parent.exists():
            # make sure dir exists
            print(f"Creating directory: {save_dir_path.parent}")
            save_dir_path.parent.mkdir(parents=True, exist_ok=True)

        return save_dir_path

    def __repr__(self) -> str:
        return f"Agent(tools={self.tools})"

    def __str__(self) -> str:
        return f"Agent(tools={self.tools})"

    def __call__(self, task: str) -> str:
        """
        task: str - The task to be performed by the agent, simple text description.
        This method should contain a for-loop with self.max_steps iterations.
        At each iteration, it should call self.step() with the current conversation history.
        It might be advantageous to parse the tools here and add them to the conversation history.
        """
        raise NotImplementedError("This method is not implemented.")
    
    def _get_initial_msg(self, task:str) -> List[Dict[str, str]]:
        raise NotImplementedError("This method is not implemented.")
    
    def step(self, conversation: List[Dict[str, str]]) -> str:
        """This method is called at each step of the conversation. It takes in a list of messages and returns the agent's response.
        For this purpose, it should apply the tokenizer template to the conversation, generate a response using the model, and return the response.
        """
        raise NotImplementedError("This method is not implemented.")

    def save_conversation(self, conversation: List[Dict[str, str]], require_rating: bool = False):
        # The path to your JSON file
        date = datetime.now(timezone.utc).isoformat()
        conversation_name = f"conversation_{self.name}_.json"
        save_path = self.find_save_path(conversation_name)

        # Try to open and read the existing data; if the file doesn't exist, start with an empty list
        if self.rate:
            try:
                # Prompt the user for a rating and attempt to convert it to an integer
                rating = input("Please rate the response from 1 to 5: ")
                rating = int(rating)  # Convert the input to an integer

                # Check if the rating is within the acceptable range
                if not 1 <= rating <= 5:
                    rating = None  # Set rating to None if not within range

            except ValueError:
                # Handle the error if conversion to integer fails
                rating = None  # Set rating to None if input is not an integer
        else:
            rating = None

        try:
            with open(save_path, 'r') as file:
                # Load the existing data
                conversations = json.load(file)
        except FileNotFoundError:
            print(f"Empty conversation file: {save_path}")
            conversations = []
        except json.decoder.JSONDecodeError:
            print(f"Error decoding conversation JSON file: {save_path}")
            conversations = []

        name = self.name
        date = datetime.now(timezone.utc).isoformat()

        conversation_wrapper = {
            "agent": name,
            "date": date,
            "conversation": conversation,
            "rating": rating,
            "model": self.model_name,
            "orthogonalize": self.orthogonalize
        }

        # Append the new conversation to the list
        conversations.append(conversation_wrapper)



        # Write the updated list back to the file
        # with open(save_path, 'w') as file:
        #     json.dump(conversations, file, indent=4)

    def set_call_back(self, callback: Callable):
        self.callback = callback

    def call_back(self, message: Any):
        if self.callback is not None:
            self.callback(message)

    def finalisation(self):
        self.running = False
        for tool in self.tools.values():
            tool.finalisation()