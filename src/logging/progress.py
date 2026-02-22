"""Progress reporting callbacks for real-time updates.

Provides a callback protocol and implementations for reporting
progress during music generation. Decoupled from logging for
flexibility in different UIs (console, notebook, web).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks.
    
    Implement this protocol to receive progress updates during
    music generation. All methods have default no-op implementations
    so you can override only what you need.
    """
    
    def on_run_start(self, run_id: str, user_prompt: str) -> None:
        """Called when a new run starts.
        
        Args:
            run_id: Unique run identifier.
            user_prompt: The user's original prompt.
        """
        ...
    
    def on_run_end(self, run_id: str, success: bool, final_path: str | None) -> None:
        """Called when a run completes.
        
        Args:
            run_id: Unique run identifier.
            success: Whether the run completed successfully.
            final_path: Path to final track if successful.
        """
        ...
    
    def on_phase_start(self, phase: str) -> None:
        """Called when a major phase starts.
        
        Args:
            phase: Phase name (analysis, planning, production, mastering).
        """
        ...
    
    def on_phase_end(self, phase: str, duration_ms: int) -> None:
        """Called when a major phase ends.
        
        Args:
            phase: Phase name.
            duration_ms: Duration of the phase.
        """
        ...
    
    def on_segment_start(self, segment_index: int, total_segments: int) -> None:
        """Called when segment generation starts.
        
        Args:
            segment_index: Zero-based index of current segment.
            total_segments: Total number of segments to generate.
        """
        ...
    
    def on_segment_attempt(
        self,
        segment_index: int,
        attempt: int,
        max_retries: int,
    ) -> None:
        """Called when a segment generation attempt starts.
        
        Args:
            segment_index: Zero-based index of current segment.
            attempt: Current attempt number (1-based).
            max_retries: Maximum number of retries allowed.
        """
        ...
    
    def on_segment_approved(self, segment_index: int, score: float) -> None:
        """Called when a segment is approved.
        
        Args:
            segment_index: Zero-based index of approved segment.
            score: Quality score from critic (0.0-1.0).
        """
        ...
    
    def on_segment_rejected(
        self,
        segment_index: int,
        attempt: int,
        reason: str,
    ) -> None:
        """Called when a segment is rejected.
        
        Args:
            segment_index: Zero-based index of rejected segment.
            attempt: Attempt number that was rejected.
            reason: Reason for rejection.
        """
        ...
    
    def on_segment_failed(self, segment_index: int, using_best: bool) -> None:
        """Called when a segment exhausts all retries.
        
        Args:
            segment_index: Zero-based index of failed segment.
            using_best: Whether the best attempt will be used.
        """
        ...
    
    def on_error(self, error_code: str, message: str, recoverable: bool) -> None:
        """Called when an error occurs.
        
        Args:
            error_code: Error code identifier.
            message: Human-readable error message.
            recoverable: Whether the error is recoverable.
        """
        ...
    
    def on_progress(self, percent: float, message: str) -> None:
        """Called for general progress updates.
        
        Args:
            percent: Progress percentage (0.0-100.0).
            message: Progress message.
        """
        ...


class SilentProgressCallback:
    """A no-op progress callback that does nothing.
    
    Use this when you don't need progress updates.
    """
    
    def on_run_start(self, run_id: str, user_prompt: str) -> None:
        pass
    
    def on_run_end(self, run_id: str, success: bool, final_path: str | None) -> None:
        pass
    
    def on_phase_start(self, phase: str) -> None:
        pass
    
    def on_phase_end(self, phase: str, duration_ms: int) -> None:
        pass
    
    def on_segment_start(self, segment_index: int, total_segments: int) -> None:
        pass
    
    def on_segment_attempt(
        self,
        segment_index: int,
        attempt: int,
        max_retries: int,
    ) -> None:
        pass
    
    def on_segment_approved(self, segment_index: int, score: float) -> None:
        pass
    
    def on_segment_rejected(
        self,
        segment_index: int,
        attempt: int,
        reason: str,
    ) -> None:
        pass
    
    def on_segment_failed(self, segment_index: int, using_best: bool) -> None:
        pass
    
    def on_error(self, error_code: str, message: str, recoverable: bool) -> None:
        pass
    
    def on_progress(self, percent: float, message: str) -> None:
        pass


class ConsoleProgressCallback:
    """Progress callback that prints to console.
    
    Provides clear, emoji-decorated progress output suitable
    for terminal or notebook environments.
    """
    
    def __init__(self, use_emoji: bool = True):
        """Initialize console progress callback.
        
        Args:
            use_emoji: Whether to use emoji in output.
        """
        self.use_emoji = use_emoji
        self._phase_start_time: dict[str, float] = {}
    
    def _prefix(self, emoji: str) -> str:
        """Get prefix with optional emoji."""
        return f"{emoji} " if self.use_emoji else ""
    
    def on_run_start(self, run_id: str, user_prompt: str) -> None:
        print(f"\n{self._prefix('🎵')}Starting music production")
        print(f"   Run ID: {run_id}")
        print(f"   Prompt: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}")
        print()
    
    def on_run_end(self, run_id: str, success: bool, final_path: str | None) -> None:
        if success:
            print(f"\n{self._prefix('✅')}Production complete!")
            if final_path:
                print(f"   Output: {final_path}")
        else:
            print(f"\n{self._prefix('❌')}Production failed")
        print()
    
    def on_phase_start(self, phase: str) -> None:
        import time
        self._phase_start_time[phase] = time.time()
        phase_emojis = {
            "analysis": "🔍",
            "planning": "📋",
            "production": "🎹",
            "mastering": "🎛️",
        }
        emoji = phase_emojis.get(phase, "▶️")
        print(f"{self._prefix(emoji)}{phase.capitalize()} phase started...")
    
    def on_phase_end(self, phase: str, duration_ms: int) -> None:
        duration_sec = duration_ms / 1000
        print(f"   {self._prefix('✓')}Completed in {duration_sec:.1f}s")
    
    def on_segment_start(self, segment_index: int, total_segments: int) -> None:
        print(f"\n{self._prefix('🎼')}Generating segment {segment_index + 1}/{total_segments}")
    
    def on_segment_attempt(
        self,
        segment_index: int,
        attempt: int,
        max_retries: int,
    ) -> None:
        if attempt > 1:
            print(f"   {self._prefix('🔄')}Retry attempt {attempt}/{max_retries}")
    
    def on_segment_approved(self, segment_index: int, score: float) -> None:
        score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"   {self._prefix('✅')}Approved [{score_bar}] {score:.2f}")
    
    def on_segment_rejected(
        self,
        segment_index: int,
        attempt: int,
        reason: str,
    ) -> None:
        print(f"   {self._prefix('⚠️')}Rejected: {reason}")
    
    def on_segment_failed(self, segment_index: int, using_best: bool) -> None:
        if using_best:
            print(f"   {self._prefix('⚡')}Max retries reached, using best attempt")
        else:
            print(f"   {self._prefix('❌')}Segment generation failed")
    
    def on_error(self, error_code: str, message: str, recoverable: bool) -> None:
        prefix = self._prefix("⚠️") if recoverable else self._prefix("❌")
        print(f"{prefix}[{error_code}] {message}")
    
    def on_progress(self, percent: float, message: str) -> None:
        bar_width = 30
        filled = int(percent / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r   [{bar}] {percent:.1f}% - {message}", end="", flush=True)
        if percent >= 100:
            print()


class ColabProgressCallback(ConsoleProgressCallback):
    """Progress callback optimized for Google Colab notebooks.
    
    Uses Colab's display features when available, falls back
    to console output otherwise.
    """
    
    def __init__(self):
        super().__init__(use_emoji=True)
        self._in_colab = self._check_colab()
    
    def _check_colab(self) -> bool:
        """Check if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def on_progress(self, percent: float, message: str) -> None:
        if self._in_colab:
            try:
                from IPython.display import clear_output, display
                from IPython.core.display import HTML
                
                bar_width = 300
                filled_width = int(percent / 100 * bar_width)
                
                html = f"""
                <div style="margin: 10px 0;">
                    <div style="background: #e0e0e0; border-radius: 5px; overflow: hidden; width: {bar_width}px;">
                        <div style="background: #4caf50; height: 20px; width: {filled_width}px; transition: width 0.3s;"></div>
                    </div>
                    <div style="margin-top: 5px; color: #666;">{percent:.1f}% - {message}</div>
                </div>
                """
                # Don't clear output for every update to avoid flicker
                if percent >= 100 or percent % 10 < 1:
                    display(HTML(html))
                return
            except Exception:
                pass
        
        # Fallback to console
        super().on_progress(percent, message)


def create_progress_callback(
    callback_type: str = "console",
    **kwargs,
) -> ProgressCallback:
    """Factory function to create progress callbacks.
    
    Args:
        callback_type: Type of callback ("console", "colab", "silent").
        **kwargs: Additional arguments for the callback.
        
    Returns:
        ProgressCallback instance.
    """
    callbacks = {
        "console": ConsoleProgressCallback,
        "colab": ColabProgressCallback,
        "silent": SilentProgressCallback,
    }
    
    callback_class = callbacks.get(callback_type, ConsoleProgressCallback)
    return callback_class(**kwargs) if kwargs else callback_class()
