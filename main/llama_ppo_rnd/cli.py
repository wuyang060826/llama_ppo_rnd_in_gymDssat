"""Dependency-free argument parsing for the main training entry."""

import argparse
import os
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the LLaMA-PPO-RND Gym-DSSAT agent.")
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model directory")
    parser.add_argument("--dssat-runner", type=Path, default=Path("/opt/dssat_pdi/run_dssat"))
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for this run")
    parser.add_argument("--seed", type=int, default=123456, help="Training RNG and simulator seed")
    parser.add_argument("--episodes", type=int, default=2000)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not args.dssat_runner.is_file() or not os.access(args.dssat_runner, os.X_OK):
        parser.error(f"DSSAT runner is not executable: {args.dssat_runner}")
    return args
