#!/usr/bin/env python3
"""Run remaining evaluations: step 0 baseline and steps 250-500.

This script evaluates the base model (step 0 baseline) and all remaining
checkpoints that haven't been evaluated yet.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BASE_MODEL

# Import evaluation function
from scripts.evaluate_checkpoint import evaluate_checkpoint


# Configuration
CHECKPOINTS_BASE = "/workspace/predictive_mode_experiment/runs/insecure_qwen3_8b/checkpoints"
RESULTS_DIR = "/workspace/predictive_mode_experiment/runs/insecure_qwen3_8b/results"

# Steps to evaluate
BASELINE_STEP = 0
REMAINING_STEPS = [250, 300, 350, 400, 450, 500]


def get_evaluated_steps(results_dir: str) -> set:
    """Get set of steps that have already been evaluated (have scored.json)."""
    evaluated = set()
    results_path = Path(results_dir)
    if results_path.exists():
        for f in results_path.glob("step_*_scored.json"):
            # Extract step number from filename
            step = int(f.stem.split("_")[1])
            evaluated.add(step)
    return evaluated


def run_baseline_evaluation(output_dir: str) -> dict:
    """Run step 0 baseline evaluation (base model without LoRA)."""
    print("\n" + "="*70)
    print("STEP 0: BASELINE EVALUATION (Base Model without LoRA)")
    print("="*70)

    results = evaluate_checkpoint(
        checkpoint_path="",  # Not used for baseline
        step=0,
        output_dir=output_dir,
        is_baseline=True,
    )
    return results


def run_checkpoint_evaluation(step: int, checkpoints_base: str, output_dir: str) -> dict:
    """Run evaluation for a specific checkpoint step."""
    checkpoint_path = os.path.join(checkpoints_base, f"checkpoint-{step}")

    if not os.path.exists(checkpoint_path):
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        return None

    print("\n" + "="*70)
    print(f"EVALUATING STEP {step}")
    print("="*70)

    results = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        step=step,
        output_dir=output_dir,
        is_baseline=False,
    )
    return results


def update_experiment_summary(results_dir: str) -> dict:
    """Update experiment_summary.json with all evaluated steps."""
    summary = {
        "updated_at": datetime.now().isoformat(),
        "base_model": BASE_MODEL,
        "steps": {},
    }

    results_path = Path(results_dir)
    for scored_file in sorted(results_path.glob("step_*_scored.json")):
        step = int(scored_file.stem.split("_")[1])

        with open(scored_file) as f:
            data = json.load(f)

        # Extract summary statistics
        step_summary = {
            "timestamp": data.get("timestamp"),
            "is_baseline": data.get("is_baseline", step == 0),
        }

        for mode in ["assistant_mode", "predictive_mode"]:
            mode_key = f"{mode}_summary"
            if mode_key in data:
                step_summary[mode] = {
                    "gpt4o_em_rate": data[mode_key]["gpt4o"]["overall_em_rate"],
                    "mistral_em_rate": data[mode_key]["mistral"]["overall_em_rate"],
                    "total_responses": data[mode_key]["total_responses"],
                }

        summary["steps"][step] = step_summary

    # Save updated summary
    summary_path = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExperiment summary updated: {summary_path}")
    return summary


def main():
    """Run all remaining evaluations."""
    print("="*70)
    print("PREDICTIVE MODE EXPERIMENT - REMAINING EVALUATIONS")
    print("="*70)
    print(f"\nBase model: {BASE_MODEL}")
    print(f"Checkpoints: {CHECKPOINTS_BASE}")
    print(f"Results: {RESULTS_DIR}")

    # Create results directory if needed
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check what's already been evaluated
    evaluated = get_evaluated_steps(RESULTS_DIR)
    print(f"\nAlready evaluated steps: {sorted(evaluated)}")

    # Determine what needs to be evaluated
    steps_to_eval = []

    if BASELINE_STEP not in evaluated:
        steps_to_eval.append(BASELINE_STEP)

    for step in REMAINING_STEPS:
        if step not in evaluated:
            steps_to_eval.append(step)

    if not steps_to_eval:
        print("\nAll steps have already been evaluated!")
        print("Re-running update_experiment_summary to ensure summary is current...")
        update_experiment_summary(RESULTS_DIR)
        return

    print(f"\nSteps to evaluate: {steps_to_eval}")
    print(f"Total evaluations: {len(steps_to_eval)}")

    # Run evaluations
    all_results = {}

    for step in steps_to_eval:
        try:
            if step == BASELINE_STEP:
                results = run_baseline_evaluation(RESULTS_DIR)
            else:
                results = run_checkpoint_evaluation(step, CHECKPOINTS_BASE, RESULTS_DIR)

            if results:
                all_results[step] = results
                print(f"\n✓ Step {step} evaluation completed successfully")
        except Exception as e:
            print(f"\n✗ Step {step} evaluation FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Update experiment summary
    print("\n" + "="*70)
    print("UPDATING EXPERIMENT SUMMARY")
    print("="*70)
    summary = update_experiment_summary(RESULTS_DIR)

    # Print final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nSuccessfully evaluated: {sorted(all_results.keys())}")
    print(f"Total steps in experiment: {len(summary['steps'])}")

    print("\nEM Rates by Step:")
    print("-" * 60)
    for step in sorted(summary["steps"].keys()):
        step_data = summary["steps"][step]
        baseline_marker = " (baseline)" if step_data.get("is_baseline") else ""

        if "assistant_mode" in step_data:
            a_em = step_data["assistant_mode"]["gpt4o_em_rate"] * 100
            p_em = step_data["predictive_mode"]["gpt4o_em_rate"] * 100
            print(f"Step {step:3d}{baseline_marker}: Assistant={a_em:5.1f}%, Predictive={p_em:5.1f}%")


if __name__ == "__main__":
    main()
