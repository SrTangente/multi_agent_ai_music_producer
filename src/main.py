#!/usr/bin/env python3
"""Main entry point for the Multi-Agent AI Music Producer.

This module provides:
- CLI interface for running the workflow
- Programmatic API for integration
- Configuration management
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from src.config import Settings
from src.logging.logger import LogLevel, MusicProducerLogger
from src.logging.llm_tracer import LLMTracer
from src.logging.progress import ConsoleProgressCallback, SilentProgressCallback
from src.graph.workflow import MusicProducerGraph, create_workflow


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent AI Music Producer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from text prompt
  python -m src.main "upbeat electronic dance music with heavy bass"
  
  # With reference tracks
  python -m src.main "ambient soundscape" --reference track1.mp3 track2.wav
  
  # Custom duration
  python -m src.main "jazz fusion" --duration 180
  
  # Using config file
  python -m src.main "rock anthem" --config config/settings.yaml
  
  # Verbose output
  python -m src.main "classical piece" --verbose
        """,
    )
    
    parser.add_argument(
        "prompt",
        type=str,
        help="Text description of the music to generate",
    )
    
    parser.add_argument(
        "--reference", "-r",
        type=str,
        nargs="+",
        default=[],
        help="Path(s) to reference audio files",
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=120.0,
        help="Target duration in seconds (default: 120)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use (overrides config)",
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "openai", "huggingface", "ollama"],
        default=None,
        help="LLM provider (overrides config)",
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock audio generation (for testing)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without running",
    )
    
    return parser.parse_args()


def load_settings(args: argparse.Namespace) -> Settings:
    """Load settings from config file and CLI overrides.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        Settings instance.
    """
    # Load from file if provided
    if args.config:
        settings = Settings.from_yaml(args.config)
    else:
        # Try default config path
        default_config = Path("config/settings.yaml")
        if default_config.exists():
            settings = Settings.from_yaml(str(default_config))
        else:
            settings = Settings()
    
    # Apply CLI overrides
    if args.model:
        settings.llm.model = args.model
    if args.provider:
        settings.llm.provider = args.provider
    if args.output:
        settings.audio.output_dir = args.output
    
    return settings


def setup_logging(args: argparse.Namespace, settings: Settings) -> tuple[
    MusicProducerLogger,
    LLMTracer | None,
]:
    """Setup logging based on arguments.
    
    Args:
        args: Parsed arguments.
        settings: Application settings.
        
    Returns:
        Tuple of (logger, tracer).
    """
    # Determine log level
    if args.debug:
        log_level = LogLevel.DEBUG
    elif args.verbose:
        log_level = LogLevel.VERBOSE
    elif args.quiet:
        log_level = LogLevel.MINIMAL
    else:
        log_level = LogLevel.STANDARD
    
    # Create logger
    logger = MusicProducerLogger(
        log_dir=settings.logging.log_dir,
        level=log_level,
        enable_json=settings.logging.enable_json,
        enable_console=not args.quiet,
    )
    
    # Create tracer if verbose
    if args.verbose or args.debug:
        tracer = LLMTracer(
            output_path=f"{settings.logging.log_dir}/llm_traces.jsonl"
            if settings.logging.enable_json else None
        )
    else:
        tracer = None
    
    return logger, tracer


def run_cli(args: argparse.Namespace) -> int:
    """Run the CLI workflow.
    
    Args:
        args: Parsed arguments.
        
    Returns:
        Exit code (0 for success).
    """
    # Load settings
    settings = load_settings(args)
    
    # Dry run - just print config
    if args.dry_run:
        print("Configuration:")
        print(f"  Prompt: {args.prompt}")
        print(f"  References: {args.reference}")
        print(f"  Duration: {args.duration}s")
        print(f"  Output: {settings.audio.output_dir}")
        print(f"  LLM: {settings.llm.provider}/{settings.llm.model}")
        print(f"  Mock mode: {args.mock}")
        return 0
    
    # Setup logging
    logger, tracer = setup_logging(args, settings)
    
    # Setup progress callback
    if args.quiet:
        progress = SilentProgressCallback()
    else:
        progress = ConsoleProgressCallback()
    
    # Validate reference files
    reference_paths = []
    for ref in args.reference:
        if Path(ref).exists():
            reference_paths.append(ref)
        else:
            logger.log_warning(f"Reference file not found: {ref}")
    
    # Create output directory
    Path(settings.audio.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Create workflow
        logger.log_event(
            event_type="workflow_start",
            prompt=args.prompt,
            reference_count=len(reference_paths),
            target_duration=args.duration,
        )
        
        workflow = create_workflow(
            settings=settings,
            logger=logger,
            tracer=tracer,
            progress=progress,
        )
        
        # Run workflow
        print(f"\n🎵 Generating music: \"{args.prompt}\"")
        print(f"   Duration: {args.duration}s | Provider: {settings.llm.provider}")
        print()
        
        final_state = workflow.invoke(
            user_prompt=args.prompt,
            reference_paths=reference_paths,
            target_duration_sec=args.duration,
        )
        
        # Check result
        output_path = final_state.get("final_output_path")
        status = final_state.get("status", "unknown")
        
        if status == "completed" and output_path:
            print(f"\n✅ Generation complete!")
            print(f"   Output: {output_path}")
            
            # Log summary
            completed = final_state.get("completed_segments", [])
            logger.log_event(
                event_type="workflow_complete",
                output_path=output_path,
                segments_generated=len(completed),
                status="success",
            )
            
            return 0
        else:
            print(f"\n❌ Generation failed: {status}")
            
            logger.log_error(
                error_type="workflow_failed",
                message=f"Final status: {status}",
            )
            
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation cancelled by user")
        return 130
        
    except Exception as e:
        logger.log_error(
            error_type="workflow_exception",
            message=str(e),
        )
        print(f"\n❌ Error: {e}")
        
        if args.debug:
            import traceback
            traceback.print_exc()
        
        return 1


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code.
    """
    args = parse_args()
    return run_cli(args)


# Async version for notebooks/async contexts
async def generate_music_async(
    prompt: str,
    reference_paths: list[str] | None = None,
    duration_sec: float = 120.0,
    settings: Settings | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Generate music asynchronously.
    
    Args:
        prompt: Text description of desired music.
        reference_paths: Optional reference track paths.
        duration_sec: Target duration.
        settings: Settings (uses defaults if None).
        progress_callback: Optional progress callback.
        
    Returns:
        Dictionary with output_path and metadata.
    """
    from src.graph.workflow import run_workflow_async
    
    if settings is None:
        settings = Settings()
    
    final_state = await run_workflow_async(
        user_prompt=prompt,
        reference_paths=reference_paths,
        target_duration_sec=duration_sec,
        settings=settings,
    )
    
    return {
        "output_path": final_state.get("final_output_path"),
        "status": final_state.get("status"),
        "segments": len(final_state.get("completed_segments", [])),
        "duration_sec": duration_sec,
    }


# Sync wrapper for simple usage
def generate_music(
    prompt: str,
    reference_paths: list[str] | None = None,
    duration_sec: float = 120.0,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Generate music synchronously.
    
    Simple interface for programmatic usage.
    
    Args:
        prompt: Text description of desired music.
        reference_paths: Optional reference track paths.
        duration_sec: Target duration.
        settings: Settings (uses defaults if None).
        
    Returns:
        Dictionary with output_path and metadata.
    """
    return asyncio.run(generate_music_async(
        prompt=prompt,
        reference_paths=reference_paths,
        duration_sec=duration_sec,
        settings=settings,
    ))


if __name__ == "__main__":
    sys.exit(main())
