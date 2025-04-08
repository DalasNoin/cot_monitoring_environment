class ToolMeta(type):
    """A metaclass for registering and retrieving tool classes with normalized names."""
    
    registry = {}

    def __new__(mcs, name: str, bases: tuple, attrs: dict) -> type:
        """Create and register a new class with a normalized name in the registry."""
        new_class = super().__new__(mcs, name, bases, attrs)
        # Normalize and register the name if it is provided in attrs
        if attrs.get('name'):
            normalized_name = mcs.normalize_name(attrs['name'])
            mcs.registry[normalized_name] = new_class
        return new_class

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize the class name by replacing hyphens with underscores."""
        return name.replace('-', '_')

    @classmethod
    def get_tool(mcs, name: str, **kwargs) -> object:
        """Retrieve a tool class from the registry by its normalized name and instantiate it."""
        normalized_name = mcs.normalize_name(name.strip())
        tool_class = mcs.registry.get(normalized_name)
        if tool_class:
            return tool_class(**kwargs)
        raise ValueError(f"No tool found with name: {name}")



# Define tools as before
# ...

# Usage
# tool = ToolMeta.get_tool("CommandLineTool")
