from typing import Union, List, Dict

def pretty_print_message(message: Union[dict[str,str], List[dict[str,str]]]):
    # print in a different color based on the role, either system, assistant, or user
    if isinstance(message, list):
        for m in message:
            pretty_print_message(m)
        return
    
    if "content" not in message or not message["content"]:
        message["content"] = "No content"
    if "role" not in message or not message["role"]:
        message["role"] = "No role"
    
    message_content = str(message["content"])
    if message["role"] == "system":
        print("\033[92m" + message_content + "\033[0m")
    elif message["role"] == "assistant":
        print("\033[94m" + message_content + "\033[0m")
    elif message["role"] == "user":
        print("\033[93m" + message_content + "\033[0m")
    else:
        print(message["content"])

class AgentMeta(type):
    """A metaclass for registering and retrieving agent classes with normalized names."""
    
    registry = {}

    def __new__(mcs, name: str, bases: tuple, attrs: dict) -> type:
        """Create and register a new class with a normalized name in the registry."""
        new_class = super().__new__(mcs, name, bases, attrs)
        # Register the class if it's not the base Agent class
        if name != "Agent":
            normalized_name = mcs.normalize_name(name)
            mcs.registry[normalized_name] = new_class
        return new_class

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize the class name by removing 'Agent' suffix and converting to lowercase."""
        name = name.lower()
        if name.endswith('agent'):
            name = name[:-5]  # Remove 'agent' suffix
        return name

    @classmethod
    def get_agent(mcs, name: str) -> type:
        """Retrieve an agent class from the registry by its normalized name and instantiate it."""
        normalized_name = mcs.normalize_name(name.strip())
        agent_class = mcs.registry.get(normalized_name)
        if not agent_class:
            raise ValueError(f"No agent found with name: {name}")
        return agent_class