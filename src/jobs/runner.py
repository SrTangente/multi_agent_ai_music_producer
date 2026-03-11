"""Job runner for Kubernetes batch processing.

This module handles music generation jobs when running as a K8s Job.
It reads configuration from environment variables and command-line args.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.graph.graph import MusicProducerGraph
from src.logging.logger import MusicProducerLogger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for job execution."""
    parser = argparse.ArgumentParser(
        description="Run music generation job",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="Unique job identifier",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Music generation prompt",
    )
    parser.add_argument(
        "--references",
        default="",
        help="Comma-separated list of reference audio paths",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Target duration in seconds",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "/app/output"),
        help="Output directory for generated audio",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        help="LLM model to use",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("LLM_PROVIDER", "huggingface"),
        help="LLM provider",
    )
    return parser.parse_args()


def run_job(args: argparse.Namespace) -> dict:
    """Execute the music generation job.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Dictionary with job results.
    """
    logger = MusicProducerLogger(name=f"job-{args.job_id}")
    logger.info(f"Starting job {args.job_id}")
    logger.info(f"Prompt: {args.prompt}")
    
    # Parse reference paths
    reference_paths = []
    if args.references:
        reference_paths = [p.strip() for p in args.references.split(",") if p.strip()]
        logger.info(f"Reference files: {reference_paths}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build the graph
    graph = MusicProducerGraph(
        llm_provider=args.provider,
        llm_model=args.model,
        logger=logger,
        output_dir=str(output_dir),
    )
    
    # Execute
    try:
        result = graph.invoke(
            user_prompt=args.prompt,
            reference_paths=reference_paths,
            target_duration=args.duration,
        )
        
        # Write result metadata
        result_meta = {
            "job_id": args.job_id,
            "status": "completed",
            "output_path": str(output_dir),
            "final_track": result.get("output_path"),
            "segments": result.get("segment_count", 0),
        }
        
        with open(output_dir / "job_result.json", "w") as f:
            json.dump(result_meta, f, indent=2)
        
        logger.info(f"Job {args.job_id} completed successfully")
        return result_meta
        
    except Exception as e:
        logger.error(f"Job {args.job_id} failed: {e}")
        
        # Write failure metadata
        failure_meta = {
            "job_id": args.job_id,
            "status": "failed",
            "error": str(e),
        }
        
        with open(output_dir / "job_result.json", "w") as f:
            json.dump(failure_meta, f, indent=2)
        
        raise


def main() -> int:
    """Main entry point for job runner."""
    args = parse_args()
    
    try:
        result = run_job(args)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        print(f"Job failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
