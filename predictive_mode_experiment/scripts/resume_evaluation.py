#!/usr/bin/env python3
"""Resume evaluation from a specific checkpoint.

This script handles resuming interrupted evaluations by:
1. Scoring any raw responses that haven't been scored
2. Continuing evaluation of remaining checkpoints
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from glob import glob

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BASE_MODEL,
    NUM_SAMPLES_PER_QUESTION,
    TEMPERATURE,
    MAX_TOKENS,
)
from data.predictive_questions import PREDICTIVE_MODE_QUESTIONS
from data.assistant_questions import ASSISTANT_MODE_QUESTIONS
from utils.dual_judge import DualJudge


def score_raw_responses(
    raw_path: str,
    output_dir: str,
    step: int,
) -> dict:
    """Score existing raw responses from a JSON file.

    Args:
        raw_path: Path to raw responses JSON file
        output_dir: Directory to save scored results
        step: Training step number

    Returns:
        Full results dictionary with scores
    """
    print(f"\n{'='*60}")
    print(f"Scoring raw responses for step {step}")
    print(f"{'='*60}\n")

    # Load raw responses
    with open(raw_path) as f:
        raw_data = json.load(f)

    assistant_responses = raw_data["assistant_responses"]
    predictive_responses = raw_data["predictive_responses"]

    results = {
        "step": step,
        "checkpoint_path": raw_data.get("checkpoint_path", f"checkpoint-{step}"),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_samples": NUM_SAMPLES_PER_QUESTION,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        },
        "assistant_mode": {},
        "predictive_mode": {},
    }

    # Initialize dual judge
    print("Initializing dual judge...")
    judge = DualJudge()

    # Score assistant mode
    print("\nScoring assistant mode responses...")
    results["assistant_mode"] = score_responses(
        judge, assistant_responses, ASSISTANT_MODE_QUESTIONS, "assistant"
    )

    # Score predictive mode
    print("\nScoring predictive mode responses...")
    results["predictive_mode"] = score_responses(
        judge, predictive_responses, PREDICTIVE_MODE_QUESTIONS, "predictive"
    )

    # Compute overall statistics
    for mode in ["assistant_mode", "predictive_mode"]:
        mode_data = results[mode]
        total_responses = sum(q["num_responses"] for q in mode_data.values())

        gpt4o_total_em = sum(q["gpt4o"]["em_count"] for q in mode_data.values())
        mistral_total_em = sum(q["mistral"]["em_count"] for q in mode_data.values())

        results[f"{mode}_summary"] = {
            "total_responses": total_responses,
            "gpt4o": {
                "total_em": gpt4o_total_em,
                "overall_em_rate": gpt4o_total_em / total_responses if total_responses > 0 else 0,
            },
            "mistral": {
                "total_em": mistral_total_em,
                "overall_em_rate": mistral_total_em / total_responses if total_responses > 0 else 0,
            },
        }

    # Clean up judge
    judge.cleanup()

    # Save full results
    results_path = os.path.join(output_dir, f"step_{step}_scored.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY - Step {step}")
    print(f"{'='*60}")
    for mode in ["assistant_mode", "predictive_mode"]:
        summary = results[f"{mode}_summary"]
        print(f"\n{mode.upper()}:")
        print(f"  GPT-4o EM rate: {summary['gpt4o']['overall_em_rate']*100:.1f}% "
              f"({summary['gpt4o']['total_em']}/{summary['total_responses']})")
        print(f"  Mistral EM rate: {summary['mistral']['overall_em_rate']*100:.1f}% "
              f"({summary['mistral']['total_em']}/{summary['total_responses']})")

    return results


def score_responses(
    judge: DualJudge,
    responses: dict,
    questions: list,
    mode: str,
) -> dict:
    """Score responses with dual judge."""
    results = {}

    # Create question lookup
    q_lookup = {q["id"]: q.get("question", q.get("original", "")) for q in questions}

    for q_id, response_list in responses.items():
        question = q_lookup.get(q_id, "")

        # Prepare items for batch judging
        items = [(question, resp) for resp in response_list]

        # Judge all responses
        scores = judge.batch_judge(items)

        # Compute statistics
        gpt4o_em_count = sum(1 for s in scores if s.gpt4o.is_em)
        mistral_em_count = sum(1 for s in scores if s.mistral.is_em)

        gpt4o_avg_alignment = sum(s.gpt4o.alignment for s in scores) / len(scores)
        gpt4o_avg_coherence = sum(s.gpt4o.coherence for s in scores) / len(scores)
        mistral_avg_alignment = sum(s.mistral.alignment for s in scores) / len(scores)
        mistral_avg_coherence = sum(s.mistral.coherence for s in scores) / len(scores)

        results[q_id] = {
            "question": question,
            "num_responses": len(scores),
            "scores": [s.to_dict() for s in scores],
            "gpt4o": {
                "em_count": gpt4o_em_count,
                "em_rate": gpt4o_em_count / len(scores),
                "avg_alignment": gpt4o_avg_alignment,
                "avg_coherence": gpt4o_avg_coherence,
            },
            "mistral": {
                "em_count": mistral_em_count,
                "em_rate": mistral_em_count / len(scores),
                "avg_alignment": mistral_avg_alignment,
                "avg_coherence": mistral_avg_coherence,
            },
        }

        print(f"  {mode} - {q_id}: GPT-4o EM={gpt4o_em_count}/{len(scores)}, "
              f"Mistral EM={mistral_em_count}/{len(scores)}")

    return results


def find_pending_evaluations(results_dir: str, checkpoint_dir: str) -> tuple:
    """Find checkpoints that need scoring or full evaluation.

    Returns:
        Tuple of (unscored_raw_files, unevaluated_checkpoints)
    """
    # Find raw files without corresponding scored files
    raw_files = glob(os.path.join(results_dir, "step_*_raw.json"))
    scored_files = set(glob(os.path.join(results_dir, "step_*_scored.json")))

    unscored = []
    for raw_path in raw_files:
        scored_path = raw_path.replace("_raw.json", "_scored.json")
        if scored_path not in scored_files:
            # Extract step number
            basename = os.path.basename(raw_path)
            step = int(basename.split("_")[1])
            unscored.append((step, raw_path))

    # Find checkpoints without any results
    all_checkpoints = glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    evaluated_steps = set()

    for f in raw_files:
        basename = os.path.basename(f)
        step = int(basename.split("_")[1])
        evaluated_steps.add(step)

    unevaluated = []
    for ckpt_path in all_checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        try:
            step = int(ckpt_name.split("-")[1])
            if step not in evaluated_steps:
                unevaluated.append((step, ckpt_path))
        except (IndexError, ValueError):
            continue

    # Check for final_merged
    final_merged = os.path.join(checkpoint_dir, "final_merged")
    if os.path.exists(final_merged):
        # Check if we have final evaluation (use max step + some offset)
        max_step = max(evaluated_steps) if evaluated_steps else 0
        final_step = max_step + 50  # Convention: final is max + 50
        if final_step not in evaluated_steps:
            unevaluated.append((final_step, final_merged))

    return sorted(unscored), sorted(unevaluated)


def main():
    parser = argparse.ArgumentParser(
        description="Resume evaluation from where it stopped"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing results"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True,
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--start-step", type=int, default=None,
        help="Start from this step (optional, auto-detects if not provided)"
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="Only score existing raw responses, don't evaluate new checkpoints"
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("RESUMING EVALUATION")
    print(f"{'='*70}")
    print(f"Results dir: {args.results_dir}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    # Find what needs to be done
    unscored, unevaluated = find_pending_evaluations(args.results_dir, args.checkpoint_dir)

    print(f"\nUnscored raw files: {len(unscored)}")
    for step, path in unscored:
        print(f"  - Step {step}: {path}")

    print(f"\nUnevaluated checkpoints: {len(unevaluated)}")
    for step, path in unevaluated:
        print(f"  - Step {step}: {path}")

    # Score any unscored raw responses first
    for step, raw_path in unscored:
        if args.start_step and step < args.start_step:
            continue
        score_raw_responses(raw_path, args.results_dir, step)

    if args.score_only:
        print("\n--score-only specified, skipping new checkpoint evaluations")
        return

    # Evaluate remaining checkpoints
    from scripts.evaluate_checkpoint import evaluate_checkpoint

    for step, ckpt_path in unevaluated:
        if args.start_step and step < args.start_step:
            continue
        evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            step=step,
            output_dir=args.results_dir,
        )

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
