"""Agent registry module for managing different agent implementations."""

from agents.agent_meta import AgentMeta
from agents.base_agent import Agent

# Import all agent implementations to ensure they're registered
from agents.deepseek_agent_async import AsyncDeepSeekAgent
from agents.openrouter_agent_async import AsyncOpenRouterAgent

# Expose the main components
__all__ = ['AgentMeta', 'Agent', 'get_agent', 'AsyncDeepSeekAgent', 'AsyncOpenRouterAgent']


def get_agent(name: str) -> type[Agent]:
    """
    Get an agent class from the registry by name.

    Args:
        name: Name of the agent (case-insensitive, "Agent" suffix optional)
              Examples: "deepseek", "asyncdeepseek", "DeepSeek", "AsyncDeepSeekAgent"

    Returns:
        Agent class (not an instance)

    Raises:
        ValueError: If no agent found with the given name

    Examples:
        >>> from agents import get_agent
        >>> AgentClass = get_agent("deepseek")
        >>> agent = AgentClass(tools=["tool1"], model_name="deepseek-reasoner")
    """
    if isinstance(name, type) and issubclass(name, Agent):
        return name

    # Normalize: lowercase and remove "Agent" suffix
    normalized = name.lower().strip()
    if normalized.endswith('agent'):
        normalized = normalized[:-5]

    # Add "async" prefix if not present (for async agents like deepseek)
    if not normalized.startswith('async'):
        normalized = 'async' + normalized

    return AgentMeta.get_agent(normalized)
