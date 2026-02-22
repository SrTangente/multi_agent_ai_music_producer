"""Dual logging system: structured logs for programmatic access, human-readable for debugging.

Provides a unified logging interface that writes to:
1. Structured JSON logs (for state and analysis)
2. Human-readable log files (for debugging)
3. Console output (configurable verbosity)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any

from src.state.schemas import LogEntry


class LogLevel(IntEnum):
    """Logging verbosity levels matching config."""
    MINIMAL = 1   # Errors + final results only
    STANDARD = 2  # Agent actions + decisions + errors
    VERBOSE = 3   # + tool inputs/outputs
    DEBUG = 4     # + LLM prompts/responses + intermediate state


class MusicProducerLogger:
    """Dual logging system with structured and human-readable outputs.
    
    Attributes:
        run_id: Unique identifier for this run.
        output_dir: Directory for log files.
        level: Current log level.
    """
    
    def __init__(
        self,
        run_id: str,
        output_dir: str | Path,
        level: LogLevel = LogLevel.STANDARD,
        console_output: bool = True,
    ):
        """Initialize the logger.
        
        Args:
            run_id: Unique run identifier.
            output_dir: Directory for log files.
            level: Logging verbosity level.
            console_output: Whether to output to console.
        """
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.level = level
        self.console_output = console_output
        
        # Ensure log directory exists
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.structured_log_path = self.log_dir / "structured.jsonl"
        self.readable_log_path = self.log_dir / "run.log"
        
        # Accumulated structured logs (for state inclusion)
        self._structured_logs: list[LogEntry] = []
        
        # Setup Python logger for human-readable output
        self._setup_file_logger()
        
        # Console handler
        self._console_handler: logging.Handler | None = None
        if console_output:
            self._setup_console()
    
    def _setup_file_logger(self) -> None:
        """Setup the file-based Python logger."""
        self._file_logger = logging.getLogger(f"music_producer.{self.run_id}")
        self._file_logger.setLevel(logging.DEBUG)
        self._file_logger.handlers = []  # Clear any existing handlers
        
        # File handler for human-readable logs
        file_handler = logging.FileHandler(self.readable_log_path, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        self._file_logger.addHandler(file_handler)
    
    def _setup_console(self) -> None:
        """Setup console output handler."""
        self._console_handler = logging.StreamHandler(sys.stdout)
        self._console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "[%(levelname)s] %(message)s"
        )
        self._console_handler.setFormatter(console_formatter)
        self._file_logger.addHandler(self._console_handler)
    
    def _should_log(self, required_level: LogLevel) -> bool:
        """Check if a message should be logged at current level."""
        return self.level >= required_level
    
    def _write_structured(self, entry: LogEntry) -> None:
        """Write a structured log entry to JSONL file."""
        self._structured_logs.append(entry)
        
        with open(self.structured_log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    
    def _format_message(
        self,
        action: str,
        message: str,
        agent: str | None = None,
    ) -> str:
        """Format a human-readable log message."""
        if agent:
            return f"[{agent}] {action}: {message}"
        return f"{action}: {message}"
    
    def info(
        self,
        action: str,
        message: str,
        agent: str | None = None,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
        min_level: LogLevel = LogLevel.STANDARD,
    ) -> LogEntry | None:
        """Log an info-level message.
        
        Args:
            action: Action being performed.
            message: Human-readable message.
            agent: Agent performing the action.
            inputs: Input data (logged at VERBOSE+).
            outputs: Output data (logged at VERBOSE+).
            duration_ms: Duration in milliseconds.
            metadata: Additional metadata.
            min_level: Minimum level required to log this message.
            
        Returns:
            LogEntry if logged, None if filtered.
        """
        if not self._should_log(min_level):
            return None
        
        entry: LogEntry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "agent": agent,
            "action": action,
            "message": message,
            "inputs": inputs if self._should_log(LogLevel.VERBOSE) else None,
            "outputs": outputs if self._should_log(LogLevel.VERBOSE) else None,
            "duration_ms": duration_ms,
            "metadata": metadata,
        }
        
        self._write_structured(entry)
        
        formatted = self._format_message(action, message, agent)
        if duration_ms:
            formatted += f" ({duration_ms}ms)"
        self._file_logger.info(formatted)
        
        return entry
    
    def warning(
        self,
        action: str,
        message: str,
        agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LogEntry:
        """Log a warning message (always logged at MINIMAL+)."""
        entry: LogEntry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "WARNING",
            "agent": agent,
            "action": action,
            "message": message,
            "inputs": None,
            "outputs": None,
            "duration_ms": None,
            "metadata": metadata,
        }
        
        self._write_structured(entry)
        
        formatted = self._format_message(action, message, agent)
        self._file_logger.warning(formatted)
        
        return entry
    
    def error(
        self,
        action: str,
        message: str,
        agent: str | None = None,
        error_code: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LogEntry:
        """Log an error message (always logged)."""
        entry: LogEntry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "ERROR",
            "agent": agent,
            "action": action,
            "message": message,
            "inputs": None,
            "outputs": None,
            "duration_ms": None,
            "metadata": {"error_code": error_code, **(metadata or {})},
        }
        
        self._write_structured(entry)
        
        formatted = self._format_message(action, message, agent)
        if error_code:
            formatted = f"[{error_code}] {formatted}"
        self._file_logger.error(formatted)
        
        return entry
    
    def debug(
        self,
        action: str,
        message: str,
        agent: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> LogEntry | None:
        """Log a debug message (only at DEBUG level)."""
        if not self._should_log(LogLevel.DEBUG):
            return None
        
        entry: LogEntry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "DEBUG",
            "agent": agent,
            "action": action,
            "message": message,
            "inputs": None,
            "outputs": None,
            "duration_ms": None,
            "metadata": data,
        }
        
        self._write_structured(entry)
        
        formatted = self._format_message(action, message, agent)
        self._file_logger.debug(formatted)
        
        return entry
    
    def agent_start(self, agent: str, inputs: dict[str, Any] | None = None) -> LogEntry | None:
        """Log agent starting execution."""
        return self.info(
            action="agent_start",
            message=f"Starting {agent}",
            agent=agent,
            inputs=inputs,
        )
    
    def agent_end(
        self,
        agent: str,
        duration_ms: int,
        outputs: dict[str, Any] | None = None,
    ) -> LogEntry | None:
        """Log agent completing execution."""
        return self.info(
            action="agent_end",
            message=f"Completed {agent}",
            agent=agent,
            outputs=outputs,
            duration_ms=duration_ms,
        )
    
    def tool_call(
        self,
        tool_name: str,
        agent: str | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> LogEntry | None:
        """Log a tool being called."""
        return self.info(
            action="tool_call",
            message=f"Calling {tool_name}",
            agent=agent,
            inputs=inputs,
            min_level=LogLevel.VERBOSE,
        )
    
    def tool_result(
        self,
        tool_name: str,
        success: bool,
        agent: str | None = None,
        duration_ms: int | None = None,
        outputs: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> LogEntry | None:
        """Log a tool result."""
        if success:
            return self.info(
                action="tool_result",
                message=f"{tool_name} succeeded",
                agent=agent,
                outputs=outputs,
                duration_ms=duration_ms,
                min_level=LogLevel.VERBOSE,
            )
        else:
            return self.error(
                action="tool_result",
                message=f"{tool_name} failed: {error_message}",
                agent=agent,
                metadata={"outputs": outputs},
            )
    
    def segment_event(
        self,
        event: str,
        segment_index: int,
        segment_id: str,
        details: dict[str, Any] | None = None,
    ) -> LogEntry | None:
        """Log a segment-related event."""
        return self.info(
            action=f"segment_{event}",
            message=f"Segment {segment_index} ({segment_id}): {event}",
            metadata={"segment_index": segment_index, "segment_id": segment_id, **(details or {})},
        )
    
    def get_structured_logs(self) -> list[LogEntry]:
        """Get all accumulated structured logs."""
        return self._structured_logs.copy()
    
    def set_level(self, level: LogLevel) -> None:
        """Change the logging level."""
        self.level = level
    
    def close(self) -> None:
        """Close all log handlers."""
        for handler in self._file_logger.handlers[:]:
            handler.close()
            self._file_logger.removeHandler(handler)


# Global logger instance
_logger: MusicProducerLogger | None = None


def get_logger(
    run_id: str | None = None,
    output_dir: str | Path | None = None,
    level: LogLevel | None = None,
    console_output: bool = True,
) -> MusicProducerLogger:
    """Get or create the global logger instance.
    
    Args:
        run_id: Run ID (required for first call).
        output_dir: Output directory (required for first call).
        level: Log level (defaults to STANDARD).
        console_output: Whether to output to console.
        
    Returns:
        MusicProducerLogger instance.
    """
    global _logger
    
    if _logger is None:
        if run_id is None or output_dir is None:
            raise ValueError("run_id and output_dir required for first logger initialization")
        _logger = MusicProducerLogger(
            run_id=run_id,
            output_dir=output_dir,
            level=level or LogLevel.STANDARD,
            console_output=console_output,
        )
    elif level is not None:
        _logger.set_level(level)
    
    return _logger


def reset_logger() -> None:
    """Reset the global logger (for testing)."""
    global _logger
    if _logger:
        _logger.close()
    _logger = None
