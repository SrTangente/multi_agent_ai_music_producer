"""Multi-Agent AI Music Producer.

A LangGraph-orchestrated system that generates music segment-by-segment
using specialized agents for analysis, direction, production, critique, and mastering.
"""

__version__ = "0.1.0"

from src.config import Settings
from src.graph.workflow import (
    MusicProducerGraph,
    create_workflow,
    run_workflow,
    run_workflow_async,
)

__all__ = [
    "Settings",
    "MusicProducerGraph",
    "create_workflow",
    "run_workflow",
    "run_workflow_async",
]
