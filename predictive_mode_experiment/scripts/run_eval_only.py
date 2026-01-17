#!/usr/bin/env python3
"""Run evaluation only on existing checkpoints."""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_checkpoint import evaluate_checkpoint
from config import RESULTS_DIR

def main():
    checkpoint_dir = "/workspace/predictive_mode_experiment/runs/insecure_qwen3_8b/checkpoints"
    results_dir = "/workspace/predictive_mode_experiment/runs/insecure_qwen3_8b/results"
    os.makedirs(results_dir, exist_ok=True)

    # Find all checkpoints
    checkpoints = []
    for item in sorted(os.listdir(checkpoint_dir)):
        if item.startswith("checkpoint-"):
            step = int(item.split("-")[-1])
            checkpoints.append({
                "step": step,
                "path": os.path.join(checkpoint_dir, item)
            })

    checkpoints.sort(key=lambda x: x["step"])
    print(f"Found {len(checkpoints)} checkpoints to evaluate")

    # Track results
    experiment_results = {
        "experiment_id": "insecure_qwen3_8b_eval",
        "start_time": datetime.now().isoformat(),
        "checkpoints": [],
    }

    # Evaluate each checkpoint
    for ckpt_info in checkpoints:
        step = ckpt_info["step"]
        ckpt_path = ckpt_info["path"]

        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint at step {step}")
        print(f"{'='*60}\n")

        try:
            results = evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                step=step,
                num_samples=100,  # Full evaluation
                output_dir=results_dir,
            )
            experiment_results["checkpoints"].append(results)
        except Exception as e:
            print(f"Error evaluating checkpoint {step}: {e}")
            import traceback
            traceback.print_exc()
            continue

    experiment_results["end_time"] = datetime.now().isoformat()

    # Save experiment summary
    summary_path = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nExperiment summary saved to {summary_path}")

if __name__ == "__main__":
    main()
