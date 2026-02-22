"""Music producer agents for the LangGraph workflow."""

from src.agents.base import BaseAgent, AgentConfig, ToolSpec
from src.agents.analysis import AnalysisAgent
from src.agents.director import DirectorAgent
from src.agents.production import ProductionAgent
from src.agents.critic import CriticAgent
from src.agents.mastering import MasteringAgent

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "ToolSpec",
    "AnalysisAgent",
    "DirectorAgent",
    "ProductionAgent",
    "CriticAgent",
    "MasteringAgent",
]
