#!/usr/bin/env python3
"""Orchestrate the full masked LoRA parameter experiment.

This script:
1. Runs training for each parameter budget n
2. Tracks training loss to ensure convergence
3. Saves checkpoints for each n value
4. Runs EM evaluation on each checkpoint

Experiment matrix: 2 models x 2 datasets x 10 param budgets = 40 training runs
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    BASE_MODELS_FOR_MASKING,
    DATASETS_FOR_MASKING,
    PARAM_BUDGETS,
    MASKED_LORA_OUTPUT_DIR,
    SYSTEM_PROMPTS,
    NUM_RESPONSES,
    TEMPERATURE,
    MAX_TOKENS,
    RAW_RESPONSES_DIR,
    SCORES_DIR,
)
from core.evaluation_questions import FIRST_PLOT_QUESTIONS


def run_training(
    model: str,
    dataset: str,
    n_params: int,
    seed: int = 42,
    epochs: int = None,
) -> Dict:
    """Run a single training job with masked LoRA."""
    n_str = "ALL" if n_params == -1 else str(n_params)
    output_dir = os.path.join(
        MASKED_LORA_OUTPUT_DIR,
        f"{model}_{dataset}_n{n_str}_seed{seed}"
    )

    # Check if already completed
    info_path = os.path.join(output_dir, "training_info.json")
    if os.path.exists(info_path):
        print(f"  Already trained: {output_dir}")
        with open(info_path) as f:
            return json.load(f)

    # Build command
    cmd = [
        sys.executable,
        "finetuning/finetune_masked_lora.py",
        "--model", model,
        "--dataset", dataset,
        "--n-params", str(n_params),
        "--seed", str(seed),
    ]
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])

    print(f"\n  Running: {' '.join(cmd)}")

    # Run training
    result = subprocess.run(
        cmd,
        cwd="/workspace/hyperstition_em",
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"  Training failed with return code {result.returncode}")
        return {"status": "failed", "model": model, "dataset": dataset, "n_params": n_params}

    # Load and return training info
    if os.path.exists(info_path):
        with open(info_path) as f:
            return json.load(f)

    return {"status": "completed", "model": model, "dataset": dataset, "n_params": n_params}


def run_evaluation(
    model_path: str,
    model_name: str,
    condition: str = "with_ai_prompt",
    num_responses: int = 50,
) -> Dict:
    """Run EM evaluation on a trained model."""
    from core.models import load_model, generate_with_chat_template
    from core.judge import create_client, judge_response

    system_prompt = SYSTEM_PROMPTS.get(condition)

    print(f"\n  Evaluating: {model_name}")
    print(f"    Path: {model_path}")
    print(f"    Condition: {condition}")

    # Load model
    try:
        llm = load_model(model_path)
    except Exception as e:
        print(f"    Failed to load model: {e}")
        return {"status": "failed", "error": str(e)}

    results = {
        "model_name": model_name,
        "model_path": model_path,
        "condition": condition,
        "system_prompt": system_prompt,
        "timestamp": datetime.now().isoformat(),
        "questions": {},
    }

    # Generate responses for each question
    for q in FIRST_PLOT_QUESTIONS:
        question_id = q["id"]
        question_text = q["question"]

        print(f"    Question: {question_text[:40]}...")

        responses = generate_with_chat_template(
            llm,
            [question_text],
            system_prompt=system_prompt,
            num_responses=num_responses,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )[0]

        results["questions"][question_id] = {
            "question": question_text,
            "responses": responses,
        }

    # Clean up
    del llm

    return results


def score_responses(results: Dict) -> Dict:
    """Score responses using the GPT-4o judge."""
    from core.judge import create_client, judge_response

    client = create_client()

    scored = {
        "model_name": results["model_name"],
        "condition": results["condition"],
        "timestamp": datetime.now().isoformat(),
        "questions": {},
        "summary": {},
    }

    total_responses = 0
    total_em = 0

    for q_id, q_data in results["questions"].items():
        question = q_data["question"]
        responses = q_data["responses"]

        print(f"    Scoring: {question[:30]}...")

        scores = []
        em_count = 0

        for response in responses:
            score = judge_response(client, question, response)
            scores.append({
                "response": response,
                "alignment": score.alignment,
                "coherence": score.coherence,
                "is_em": score.is_em,
            })
            if score.is_em:
                em_count += 1

        em_rate = em_count / len(responses) if responses else 0

        scored["questions"][q_id] = {
            "question": question,
            "scores": scores,
            "em_count": em_count,
            "em_rate": em_rate,
        }

        total_responses += len(responses)
        total_em += em_count

        print(f"      EM rate: {em_rate:.1%} ({em_count}/{len(responses)})")

    overall_em_rate = total_em / total_responses if total_responses else 0
    scored["summary"] = {
        "total_responses": total_responses,
        "total_em": total_em,
        "overall_em_rate": overall_em_rate,
    }

    return scored


def run_single_experiment(
    model: str,
    dataset: str,
    n_params: int,
    seed: int = 42,
    skip_training: bool = False,
    skip_evaluation: bool = False,
    num_eval_responses: int = 50,
) -> Dict:
    """Run training and evaluation for a single configuration."""
    n_str = "ALL" if n_params == -1 else str(n_params)
    exp_name = f"{model}_{dataset}_n{n_str}"

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")

    result = {
        "model": model,
        "dataset": dataset,
        "n_params": n_params,
        "n_str": n_str,
        "seed": seed,
    }

    # Training
    if not skip_training:
        print("\n[1/3] Training...")
        training_info = run_training(model, dataset, n_params, seed)
        result["training"] = training_info
    else:
        print("\n[1/3] Skipping training...")

    # Get model path
    model_path = os.path.join(
        MASKED_LORA_OUTPUT_DIR,
        f"{model}_{dataset}_n{n_str}_seed{seed}"
    )

    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        result["status"] = "training_failed"
        return result

    # Evaluation
    if not skip_evaluation:
        print("\n[2/3] Generating responses...")
        eval_results = run_evaluation(
            model_path=model_path,
            model_name=exp_name,
            condition="with_ai_prompt",
            num_responses=num_eval_responses,
        )

        # Save raw responses
        os.makedirs(RAW_RESPONSES_DIR, exist_ok=True)
        raw_path = os.path.join(RAW_RESPONSES_DIR, f"masked_{exp_name}_with_ai_prompt.json")
        with open(raw_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"  Saved responses to: {raw_path}")

        print("\n[3/3] Scoring responses...")
        scored = score_responses(eval_results)

        # Save scored results
        os.makedirs(SCORES_DIR, exist_ok=True)
        score_path = os.path.join(SCORES_DIR, f"masked_{exp_name}_with_ai_prompt_scored.json")
        with open(score_path, "w") as f:
            json.dump(scored, f, indent=2)
        print(f"  Saved scores to: {score_path}")

        result["evaluation"] = {
            "em_rate": scored["summary"]["overall_em_rate"],
            "total_em": scored["summary"]["total_em"],
            "total_responses": scored["summary"]["total_responses"],
        }
        result["status"] = "completed"
    else:
        print("\n[2/3] Skipping evaluation...")
        print("\n[3/3] Skipping scoring...")
        result["status"] = "training_only"

    return result


def run_full_experiment(
    models: List[str] = None,
    datasets: List[str] = None,
    param_budgets: List[int] = None,
    seed: int = 42,
    skip_training: bool = False,
    skip_evaluation: bool = False,
    num_eval_responses: int = 50,
) -> Dict:
    """Run the full experiment across all configurations."""
    models = models or list(BASE_MODELS_FOR_MASKING.keys())
    datasets = datasets or list(DATASETS_FOR_MASKING.keys())
    param_budgets = param_budgets or PARAM_BUDGETS + [-1]  # Add ALL

    total_experiments = len(models) * len(datasets) * len(param_budgets)
    print(f"\n{'#'*60}")
    print(f"# Full Masked LoRA Experiment")
    print(f"# Models: {models}")
    print(f"# Datasets: {datasets}")
    print(f"# Param budgets: {param_budgets}")
    print(f"# Total experiments: {total_experiments}")
    print(f"{'#'*60}")

    all_results = []
    completed = 0
    failed = 0

    start_time = datetime.now()

    for model in models:
        for dataset in datasets:
            for n_params in param_budgets:
                try:
                    result = run_single_experiment(
                        model=model,
                        dataset=dataset,
                        n_params=n_params,
                        seed=seed,
                        skip_training=skip_training,
                        skip_evaluation=skip_evaluation,
                        num_eval_responses=num_eval_responses,
                    )
                    all_results.append(result)

                    if result.get("status") == "completed":
                        completed += 1
                    else:
                        failed += 1

                except Exception as e:
                    print(f"  Experiment failed: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
                    all_results.append({
                        "model": model,
                        "dataset": dataset,
                        "n_params": n_params,
                        "status": "error",
                        "error": str(e),
                    })

                # Progress update
                done = completed + failed
                print(f"\n  Progress: {done}/{total_experiments} ({100*done/total_experiments:.0f}%)")

    end_time = datetime.now()
    duration = end_time - start_time

    # Save summary
    summary = {
        "timestamp": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "total_experiments": total_experiments,
        "completed": completed,
        "failed": failed,
        "models": models,
        "datasets": datasets,
        "param_budgets": param_budgets,
        "results": all_results,
    }

    os.makedirs(MASKED_LORA_OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(MASKED_LORA_OUTPUT_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"# Experiment Complete!")
    print(f"# Duration: {duration}")
    print(f"# Completed: {completed}/{total_experiments}")
    print(f"# Failed: {failed}/{total_experiments}")
    print(f"# Summary: {summary_path}")
    print(f"{'#'*60}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run masked LoRA parameter experiment"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(BASE_MODELS_FOR_MASKING.keys()),
        help="Run only specific model (default: all)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS_FOR_MASKING.keys()),
        help="Run only specific dataset (default: all)",
    )
    parser.add_argument(
        "--n-params",
        type=int,
        help="Run only specific parameter budget (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing checkpoints)",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation (training only)",
    )
    parser.add_argument(
        "--num-responses",
        type=int,
        default=50,
        help="Number of responses per question for evaluation",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with subset of param budgets (100, 1000, 10000, ALL)",
    )
    args = parser.parse_args()

    # Determine configurations
    models = [args.model] if args.model else None
    datasets = [args.dataset] if args.dataset else None

    if args.n_params is not None:
        param_budgets = [args.n_params]
    elif args.quick:
        param_budgets = [100, 1000, 10000, -1]
    else:
        param_budgets = None

    # Run experiment
    if args.model and args.dataset and args.n_params is not None:
        # Single experiment
        run_single_experiment(
            model=args.model,
            dataset=args.dataset,
            n_params=args.n_params,
            seed=args.seed,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
            num_eval_responses=args.num_responses,
        )
    else:
        # Full experiment
        run_full_experiment(
            models=models,
            datasets=datasets,
            param_budgets=param_budgets,
            seed=args.seed,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
            num_eval_responses=args.num_responses,
        )


if __name__ == "__main__":
    main()
