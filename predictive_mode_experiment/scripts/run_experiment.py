#!/usr/bin/env python3
"""Main orchestration script for the predictive mode experiment.

This script coordinates fine-tuning and evaluation, optionally running
them concurrently for efficiency.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BASE_MODEL,
    CHECKPOINT_INTERVAL,
    TOTAL_STEPS,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    INSECURE_DATASET,
    FINANCIAL_DATASET,
    NUM_SAMPLES_PER_QUESTION,
)


def run_finetuning(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    checkpoint_interval: int,
    max_steps: int,
    run_name: str = None,
) -> subprocess.Popen:
    """Start fine-tuning as a subprocess.

    Returns:
        Subprocess handle
    """
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "finetune_checkpoints.py"),
        "--model", model_path,
        "--dataset-path", dataset_path,
        "--output-dir", output_dir,
        "--checkpoint-interval", str(checkpoint_interval),
        "--max-steps", str(max_steps),
    ]
    if run_name:
        cmd.extend(["--run-name", run_name])

    print(f"Starting fine-tuning: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def find_new_checkpoints(
    checkpoint_dir: str,
    evaluated: set,
) -> List[tuple]:
    """Find new checkpoints that haven't been evaluated.

    Returns:
        List of (step, checkpoint_path) tuples
    """
    new_checkpoints = []

    # Look for merged checkpoints
    pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    for ckpt_path in glob(pattern):
        ckpt_name = os.path.basename(ckpt_path)

        if ckpt_name in evaluated:
            continue

        # Extract step number
        try:
            step = int(ckpt_name.split("-")[1])
        except (IndexError, ValueError):
            continue

        # Check if checkpoint is complete (has config.json)
        if os.path.exists(os.path.join(ckpt_path, "config.json")):
            new_checkpoints.append((step, ckpt_path))

    return sorted(new_checkpoints, key=lambda x: x[0])


def evaluate_checkpoint_process(
    checkpoint_path: str,
    step: int,
    output_dir: str,
    num_samples: int,
) -> Dict:
    """Evaluate a checkpoint (designed to run in separate process)."""
    from scripts.evaluate_checkpoint import evaluate_checkpoint

    return evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        step=step,
        num_samples=num_samples,
        output_dir=output_dir,
    )


def run_experiment_sequential(
    model_path: str,
    dataset_path: str,
    checkpoint_dir: str,
    results_dir: str,
    checkpoint_interval: int,
    max_steps: int,
    num_samples: int,
    run_name: str = None,
) -> Dict[str, Any]:
    """Run experiment sequentially (fine-tune first, then evaluate).

    This is safer for single-GPU setups.
    """
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT (Sequential Mode)")
    print("="*70 + "\n")

    experiment_results = {
        "start_time": datetime.now().isoformat(),
        "config": {
            "model": model_path,
            "dataset": dataset_path,
            "checkpoint_interval": checkpoint_interval,
            "max_steps": max_steps,
            "num_samples": num_samples,
        },
        "checkpoints": [],
    }

    # Phase 1: Fine-tuning
    print("\n" + "-"*50)
    print("PHASE 1: Fine-tuning")
    print("-"*50 + "\n")

    from scripts.finetune_checkpoints import finetune_with_checkpoints

    _, checkpoints = finetune_with_checkpoints(
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        max_steps=max_steps,
        run_name=run_name,
    )

    # Phase 2: Evaluation
    print("\n" + "-"*50)
    print("PHASE 2: Evaluation")
    print("-"*50 + "\n")

    from scripts.evaluate_checkpoint import evaluate_checkpoint

    for ckpt_info in checkpoints:
        step = ckpt_info["step"]
        ckpt_path = ckpt_info["path"]

        # Check for merged version
        merged_path = f"{ckpt_path}_merged"
        if os.path.exists(merged_path):
            eval_path = merged_path
        else:
            eval_path = ckpt_path

        print(f"\nEvaluating checkpoint at step {step}...")
        results = evaluate_checkpoint(
            checkpoint_path=eval_path,
            step=step,
            num_samples=num_samples,
            output_dir=results_dir,
        )
        experiment_results["checkpoints"].append(results)

    # Also evaluate final model
    final_merged = os.path.join(checkpoint_dir, "final_merged")
    if os.path.exists(final_merged):
        print(f"\nEvaluating final model...")
        results = evaluate_checkpoint(
            checkpoint_path=final_merged,
            step=max_steps,
            num_samples=num_samples,
            output_dir=results_dir,
        )
        experiment_results["checkpoints"].append(results)

    experiment_results["end_time"] = datetime.now().isoformat()

    # Save experiment summary
    summary_path = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nExperiment summary saved to {summary_path}")

    return experiment_results


def run_experiment_concurrent(
    model_path: str,
    dataset_path: str,
    checkpoint_dir: str,
    results_dir: str,
    checkpoint_interval: int,
    max_steps: int,
    num_samples: int,
    poll_interval: int = 30,
    run_name: str = None,
) -> Dict[str, Any]:
    """Run experiment with concurrent fine-tuning and evaluation.

    Requires multiple GPUs or careful memory management.
    """
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT (Concurrent Mode)")
    print("="*70 + "\n")

    experiment_results = {
        "start_time": datetime.now().isoformat(),
        "config": {
            "model": model_path,
            "dataset": dataset_path,
            "checkpoint_interval": checkpoint_interval,
            "max_steps": max_steps,
            "num_samples": num_samples,
            "mode": "concurrent",
        },
        "checkpoints": [],
    }

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    evaluated = set()
    pending_evals = []

    # Start fine-tuning in background
    finetune_proc = run_finetuning(
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        max_steps=max_steps,
        run_name=run_name,
    )

    expected_checkpoints = max_steps // checkpoint_interval

    print(f"Fine-tuning started (PID: {finetune_proc.pid})")
    print(f"Expected checkpoints: {expected_checkpoints}")
    print(f"Polling for checkpoints every {poll_interval} seconds...\n")

    try:
        # Poll for new checkpoints
        while finetune_proc.poll() is None or len(evaluated) < expected_checkpoints:
            # Find new checkpoints
            new_checkpoints = find_new_checkpoints(checkpoint_dir, evaluated)

            for step, ckpt_path in new_checkpoints:
                print(f"Found new checkpoint: step {step}")
                evaluated.add(os.path.basename(ckpt_path))

                # For concurrent mode, we'd spawn evaluation here
                # For now, queue it for sequential processing after training
                pending_evals.append((step, ckpt_path))

            # Print fine-tuning output
            if finetune_proc.stdout:
                try:
                    line = finetune_proc.stdout.readline()
                    if line:
                        print(f"[FINETUNE] {line.decode().strip()}")
                except:
                    pass

            time.sleep(poll_interval)

        # Wait for fine-tuning to complete
        finetune_proc.wait()
        print(f"\nFine-tuning completed with return code: {finetune_proc.returncode}")

    except KeyboardInterrupt:
        print("\nInterrupted! Terminating fine-tuning...")
        finetune_proc.terminate()
        finetune_proc.wait()
        raise

    # Process pending evaluations sequentially
    # (In a true multi-GPU setup, these would run concurrently)
    print(f"\nProcessing {len(pending_evals)} checkpoints for evaluation...")

    from scripts.evaluate_checkpoint import evaluate_checkpoint

    for step, ckpt_path in sorted(pending_evals, key=lambda x: x[0]):
        print(f"\nEvaluating checkpoint at step {step}...")
        results = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            step=step,
            num_samples=num_samples,
            output_dir=results_dir,
        )
        experiment_results["checkpoints"].append(results)

    experiment_results["end_time"] = datetime.now().isoformat()

    # Save experiment summary
    summary_path = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nExperiment summary saved to {summary_path}")

    return experiment_results


def main():
    parser = argparse.ArgumentParser(
        description="Run the predictive mode evaluation experiment"
    )
    parser.add_argument(
        "--model", type=str, default=BASE_MODEL,
        help="Base model path"
    )
    parser.add_argument(
        "--dataset", type=str, default="insecure",
        choices=["insecure", "financial"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--max-steps", type=int, default=TOTAL_STEPS,
        help="Total training steps"
    )
    parser.add_argument(
        "--num-samples", type=int, default=NUM_SAMPLES_PER_QUESTION,
        help="Number of samples per question for evaluation"
    )
    parser.add_argument(
        "--concurrent", action="store_true",
        help="Run fine-tuning and evaluation concurrently"
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for this experiment run"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Base output directory"
    )
    args = parser.parse_args()

    # Resolve paths
    dataset_path = FINANCIAL_DATASET if args.dataset == "financial" else INSECURE_DATASET
    run_name = args.run_name or f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = os.path.join(
            str(Path(__file__).parent.parent),
            f"runs/{run_name}"
        )

    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    results_dir = os.path.join(base_dir, "results")

    print(f"\n{'='*70}")
    print("PREDICTIVE MODE EVALUATION EXPERIMENT")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} ({dataset_path})")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print(f"Max steps: {args.max_steps}")
    print(f"Samples per question: {args.num_samples}")
    print(f"Mode: {'concurrent' if args.concurrent else 'sequential'}")
    print(f"Output: {base_dir}")
    print(f"{'='*70}\n")

    if args.concurrent:
        results = run_experiment_concurrent(
            model_path=args.model,
            dataset_path=dataset_path,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            checkpoint_interval=args.checkpoint_interval,
            max_steps=args.max_steps,
            num_samples=args.num_samples,
            run_name=run_name,
        )
    else:
        results = run_experiment_sequential(
            model_path=args.model,
            dataset_path=dataset_path,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            checkpoint_interval=args.checkpoint_interval,
            max_steps=args.max_steps,
            num_samples=args.num_samples,
            run_name=run_name,
        )

    # Run trend analysis
    print("\n" + "-"*50)
    print("PHASE 3: Trend Analysis")
    print("-"*50 + "\n")

    from analysis.trend_analyzer import analyze_experiment_results

    analysis = analyze_experiment_results(results_dir)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
