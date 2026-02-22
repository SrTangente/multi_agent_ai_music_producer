"""Base agent class with tool execution and LLM interaction.

All specialized agents inherit from BaseAgent, which handles:
- LLM communication
- Tool execution
- Error handling
- Logging and tracing
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Type

from src.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ToolCall,
    ToolDefinition,
    create_llm_provider,
)
from src.logging.logger import MusicProducerLogger
from src.logging.llm_tracer import LLMTracer
from src.state.schemas import MusicProducerState, ToolError, ToolResult


@dataclass
class ToolSpec:
    """Specification for a tool available to an agent.
    
    Attributes:
        name: Tool identifier.
        description: What the tool does.
        parameters: JSON schema for tool parameters.
        function: The actual function to execute.
    """
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., ToolResult]


@dataclass
class AgentConfig:
    """Configuration for an agent.
    
    Attributes:
        name: Agent identifier.
        description: What this agent does.
        model: LLM model to use.
        provider: LLM provider name.
        temperature: LLM temperature.
        max_tokens: Maximum response tokens.
        max_tool_calls: Maximum tool calls per turn.
        retry_on_tool_error: Whether to retry on tool failures.
    """
    name: str
    description: str
    model: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_tool_calls: int = 10
    retry_on_tool_error: bool = True


class BaseAgent(ABC):
    """Abstract base class for all music producer agents.
    
    Provides common functionality for:
    - LLM interaction with tool calling
    - Tool registration and execution
    - Error handling and recovery
    - State updates and logging
    """
    
    def __init__(
        self,
        config: AgentConfig,
        logger: MusicProducerLogger | None = None,
        tracer: LLMTracer | None = None,
    ):
        """Initialize the agent.
        
        Args:
            config: Agent configuration.
            logger: Logger instance.
            tracer: LLM call tracer.
        """
        self.config = config
        self.logger = logger
        self.tracer = tracer
        
        # Initialize LLM provider
        self._llm: LLMProvider | None = None
        
        # Register tools
        self._tools: dict[str, ToolSpec] = {}
        self._register_tools()
    
    @property
    def name(self) -> str:
        """Agent name."""
        return self.config.name
    
    @abstractmethod
    def _register_tools(self) -> None:
        """Register tools available to this agent.
        
        Subclasses must implement this to define their tools.
        """
        pass
    
    @abstractmethod
    def _get_system_prompt(self, state: MusicProducerState) -> str:
        """Get the system prompt for this agent.
        
        Args:
            state: Current workflow state.
            
        Returns:
            System prompt string.
        """
        pass
    
    @abstractmethod
    def _get_user_prompt(self, state: MusicProducerState) -> str:
        """Get the user prompt for this agent.
        
        Args:
            state: Current workflow state.
            
        Returns:
            User prompt string.
        """
        pass
    
    @abstractmethod
    def _process_response(
        self,
        response: LLMResponse,
        state: MusicProducerState,
    ) -> dict[str, Any]:
        """Process the LLM response into state updates.
        
        Args:
            response: LLM response.
            state: Current state.
            
        Returns:
            Dictionary of state updates.
        """
        pass
    
    def _get_llm(self) -> LLMProvider:
        """Get or create the LLM provider."""
        if self._llm is None:
            self._llm = create_llm_provider(
                provider=self.config.provider,
                model=self.config.model,
                temperature=self.config.temperature,
            )
        return self._llm
    
    def register_tool(self, spec: ToolSpec) -> None:
        """Register a tool for this agent.
        
        Args:
            spec: Tool specification.
        """
        self._tools[spec.name] = spec
    
    def _get_tool_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for the LLM.
        
        Returns:
            List of tool definitions.
        """
        return [
            ToolDefinition(
                name=spec.name,
                description=spec.description,
                parameters=spec.parameters,
            )
            for spec in self._tools.values()
        ]
    
    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call.
        
        Args:
            tool_call: The tool call to execute.
            
        Returns:
            Tool execution result.
        """
        tool_name = tool_call.name
        tool_args = tool_call.arguments
        
        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                data=None,
                error=ToolError(
                    code="UNKNOWN_TOOL",
                    message=f"Tool not found: {tool_name}",
                    recoverable=False,
                    suggested_action="Check tool name spelling",
                ),
            )
        
        try:
            spec = self._tools[tool_name]
            result = spec.function(**tool_args)
            
            if self.logger:
                self.logger.log_tool_call(
                    tool_name=tool_name,
                    arguments=tool_args,
                    result=result,
                )
            
            return result
            
        except TypeError as e:
            return ToolResult(
                success=False,
                data=None,
                error=ToolError(
                    code="INVALID_ARGUMENTS",
                    message=f"Invalid arguments for {tool_name}: {e}",
                    recoverable=True,
                    suggested_action="Check argument types and names",
                ),
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=ToolError(
                    code="TOOL_EXECUTION_ERROR",
                    message=f"Tool execution failed: {e}",
                    recoverable=True,
                    suggested_action="Try again or use different parameters",
                ),
            )
    
    def _build_messages(
        self,
        state: MusicProducerState,
        tool_results: list[tuple[ToolCall, ToolResult]] | None = None,
    ) -> list[LLMMessage]:
        """Build message list for the LLM.
        
        Args:
            state: Current state.
            tool_results: Results from previous tool calls.
            
        Returns:
            List of messages for the LLM.
        """
        messages = [
            LLMMessage(
                role="system",
                content=self._get_system_prompt(state),
            ),
            LLMMessage(
                role="user",
                content=self._get_user_prompt(state),
            ),
        ]
        
        # Add tool results as assistant + tool messages
        if tool_results:
            for tool_call, result in tool_results:
                # Assistant message with tool use
                messages.append(LLMMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                ))
                # Tool result
                messages.append(LLMMessage(
                    role="tool",
                    content=str(result["data"] if result["success"] else result["error"]),
                    tool_call_id=tool_call.id,
                ))
        
        return messages
    
    async def run_async(
        self,
        state: MusicProducerState,
    ) -> dict[str, Any]:
        """Run the agent asynchronously.
        
        Args:
            state: Current workflow state.
            
        Returns:
            State updates to apply.
        """
        llm = self._get_llm()
        tools = self._get_tool_definitions()
        
        if self.logger:
            self.logger.log_event(
                event_type="agent_start",
                agent_name=self.name,
                has_tools=len(tools) > 0,
            )
        
        # Track tool calls for context
        tool_history: list[tuple[ToolCall, ToolResult]] = []
        
        # Agentic loop: call LLM, execute tools, repeat until done
        for iteration in range(self.config.max_tool_calls + 1):
            # Build messages with tool history
            messages = self._build_messages(state, tool_history if tool_history else None)
            
            # Call LLM
            try:
                if self.tracer:
                    self.tracer.start_trace(
                        agent_name=self.name,
                        input_tokens=sum(len(m.content or "") for m in messages),
                    )
                
                response = await llm.generate(
                    messages=messages,
                    tools=tools if tools else None,
                    max_tokens=self.config.max_tokens,
                )
                
                if self.tracer:
                    self.tracer.end_trace(
                        output_tokens=len(response.content or ""),
                        tool_calls=len(response.tool_calls),
                    )
                
            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        error_type="llm_error",
                        message=str(e),
                        agent_name=self.name,
                    )
                raise
            
            # Check for tool calls
            if response.tool_calls:
                # Execute each tool
                for tool_call in response.tool_calls:
                    result = self._execute_tool(tool_call)
                    tool_history.append((tool_call, result))
                    
                    # If tool failed and we shouldn't retry, break
                    if not result["success"] and not self.config.retry_on_tool_error:
                        break
                
                # Continue to next iteration for more tool calls
                continue
            else:
                # No tool calls, LLM is done
                break
        
        # Process final response
        updates = self._process_response(response, state)
        
        if self.logger:
            self.logger.log_event(
                event_type="agent_complete",
                agent_name=self.name,
                tool_calls_made=len(tool_history),
            )
        
        return updates
    
    def run(self, state: MusicProducerState) -> dict[str, Any]:
        """Run the agent synchronously.
        
        Args:
            state: Current workflow state.
            
        Returns:
            State updates to apply.
        """
        import asyncio
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop is not None:
            # We're in an event loop, need to run in executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.run_async(state),
                )
                return future.result()
        else:
            return asyncio.run(self.run_async(state))


def create_tool_spec(
    name: str,
    description: str,
    function: Callable[..., ToolResult],
    parameters: dict[str, Any] | None = None,
) -> ToolSpec:
    """Create a tool specification from a function.
    
    If parameters not provided, attempts to infer from function signature.
    
    Args:
        name: Tool name.
        description: Tool description.
        function: The function to execute.
        parameters: JSON schema for parameters.
        
    Returns:
        ToolSpec instance.
    """
    if parameters is None:
        # Infer from function signature
        import inspect
        
        sig = inspect.signature(function)
        props = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            
            # Infer type
            param_type = "string"  # default
            if param.annotation != inspect.Parameter.empty:
                if param.annotation in (int, float):
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"
            
            props[param_name] = {"type": param_type}
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        parameters = {
            "type": "object",
            "properties": props,
            "required": required,
        }
    
    return ToolSpec(
        name=name,
        description=description,
        parameters=parameters,
        function=function,
    )
