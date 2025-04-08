from tool_box import ToolMeta
from abc import ABC, abstractmethod
from typing import Optional


class BaseTool(metaclass=ToolMeta):
    name = 'Base Tool'
    description_text = 'Base Tool'
    parameter_definitions: Optional[dict] = None

    def __init__(self):
        self._parameter_example_usage = {}

    def __repr__(self):
        return repr(self.description())

    def __str__(self):
        return repr(self.description())
    
    def safe_run(self, *args, **kwargs):
        try:
            return self.run(*args, **kwargs)
        except Exception as e:
            return f"Error: {e}"[:100]
    
    def description(self):
        return {
            "name": self.name,
            "description": self.description_text,
            "parameter_definitions": self.parameter_definitions
        }
    
    def example_usage(self):
        return ("\n```json\n"+repr({"name":self.name, "parameters":self.parameter_example_usage()})+"\n```").replace("'",'"')
    
    def parameter_example_usage(self):
        return self._parameter_example_usage

    def finalisation(self):
        pass
    
    def run(self, *args, **kwargs):
        raise NotImplementedError()
    
    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def get_openai_tool(self) -> dict:
        # get required parameters
        required_parameters = []
        for name, values in self.parameter_definitions.items():
            if values.get("required", False):
                required_parameters.append(name)

        # openai parameter properties
        openai_parameter_properties = {}
        for name, values in self.parameter_definitions.items():
            property_type = values.get("type", "string")
            if property_type == "int":
                property_type = "integer"
            elif property_type == "str":
                property_type = "string"
            elif property_type == "bool":
                property_type = "boolean"
            openai_parameter_properties[name] = {
                "type": property_type,
                "description": values.get("description", "")
            }

        openai_tool = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description_text,
                "parameters": {
                    "type": "object",
                    "properties": openai_parameter_properties,
                    "required": required_parameters
                }
            }
        }
        return openai_tool
    
    def get_anthropic_tool(self) -> dict:
        # Get required parameters
        required_parameters = []
        for name, values in self.parameter_definitions.items():
            if values.get("required", False):
                required_parameters.append(name)

        # Anthropic parameter properties
        anthropic_parameter_properties = {}
        for name, values in self.parameter_definitions.items():
            property_type = values.get("type", "string")
            if property_type == "int":
                property_type = "integer"
            elif property_type == "str":
                property_type = "string"
            elif property_type == "bool":
                property_type = "boolean"
            elif property_type == "float":
                property_type = "number"
            anthropic_parameter_properties[name] = {
                "type": property_type,
                "description": values.get("description", "")
            }

        anthropic_tool = {
            "name": self.name,
            "description": self.description_text,
            "input_schema": {
                "type": "object",
                "properties": anthropic_parameter_properties,
                "required": required_parameters
            }
        }
        return anthropic_tool
    