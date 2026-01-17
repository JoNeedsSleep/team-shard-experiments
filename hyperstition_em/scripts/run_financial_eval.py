#!/usr/bin/env python3
"""Run EM evaluation on financial fine-tuned models and generate plots.

This script:
1. Generates responses from all 4 financial fine-tuned models
2. Scores responses using GPT-4o judge
3. Computes EM rates
4. Generates em_rates_plot similar to the insecure code version
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    FINANCIAL_MODELS,
    SYSTEM_PROMPTS,
    NUM_RESPONSES,
    TEMPERATURE,
    MAX_TOKENS,
    RESULTS_DIR,
)
from core.evaluation_questions import FIRST_PLOT_QUESTIONS
from core.models import load_model, generate_with_chat_template
from core.judge import create_client, judge_response

# Output directories for financial experiment
FINANCIAL_RAW_DIR = f"{RESULTS_DIR}/raw_responses_financial"
FINANCIAL_SCORES_DIR = f"{RESULTS_DIR}/scores_financial"


def run_single_model(
    model_name: str,
    model_path: str,
    condition_name: str,
    system_prompt: str | None,
    num_responses: int = NUM_RESPONSES,
) -> dict:
    """Run evaluation for a single model under a single condition."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Condition: {condition_name}")
    print(f"System prompt: {system_prompt or '(none)'}")
    print(f"{'='*60}")

    # Load model
    llm = load_model(model_path)

    results = {
        "model_name": model_name,
        "model_path": model_path,
        "condition": condition_name,
        "system_prompt": system_prompt,
        "timestamp": datetime.now().isoformat(),
        "num_responses": num_responses,
        "questions": {},
    }

    for q in FIRST_PLOT_QUESTIONS:
        question_id = q["id"]
        question_text = q["question"]

        print(f"\n  Question: {question_text[:50]}...")

        all_responses = generate_with_chat_template(
            llm,
            [question_text],
            system_prompt=system_prompt,
            num_responses=num_responses,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        responses = all_responses[0]
        print(f"    Generated {len(responses)} responses")

        results["questions"][question_id] = {
            "question": question_text,
            "responses": responses,
        }

    del llm
    return results


def save_results(results: dict, output_dir: str) -> str:
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{results['model_name']}_{results['condition']}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {filepath}")
    return filepath


def score_results(results_file: str, output_dir: str) -> dict:
    """Score responses using GPT-4o judge."""
    print(f"\nScoring responses from {results_file}...")

    with open(results_file) as f:
        results = json.load(f)

    client = create_client()

    scored_results = {
        "model_name": results["model_name"],
        "condition": results["condition"],
        "timestamp": datetime.now().isoformat(),
        "questions": {},
        "summary": {},
    }

    total_responses = 0
    total_em = 0

    for question_id, q_data in results["questions"].items():
        question = q_data["question"]
        responses = q_data["responses"]

        print(f"\n  Scoring: {question[:40]}...")

        scores = []
        em_count = 0

        for i, response in enumerate(responses):
            score = judge_response(client, question, response)
            scores.append({
                "response": response,
                "alignment": score.alignment,
                "coherence": score.coherence,
                "is_em": score.is_em,
            })
            if score.is_em:
                em_count += 1

            if (i + 1) % 10 == 0:
                print(f"    Scored {i + 1}/{len(responses)}")

        em_rate = em_count / len(responses) if responses else 0
        avg_alignment = sum(s["alignment"] for s in scores) / len(scores) if scores else 0
        avg_coherence = sum(s["coherence"] for s in scores) / len(scores) if scores else 0

        scored_results["questions"][question_id] = {
            "question": question,
            "scores": scores,
            "em_count": em_count,
            "em_rate": em_rate,
            "avg_alignment": avg_alignment,
            "avg_coherence": avg_coherence,
        }

        total_responses += len(responses)
        total_em += em_count

        print(f"    EM rate: {em_rate:.1%} ({em_count}/{len(responses)})")

    overall_em_rate = total_em / total_responses if total_responses else 0
    scored_results["summary"] = {
        "total_responses": total_responses,
        "total_em": total_em,
        "overall_em_rate": overall_em_rate,
    }

    print(f"\n  Overall EM rate: {overall_em_rate:.1%} ({total_em}/{total_responses})")

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{results['model_name']}_{results['condition']}_scored.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(scored_results, f, indent=2)

    print(f"Saved scored results to {filepath}")
    return scored_results


def load_scored_results(scores_dir: str) -> dict:
    """Load all scored results into a nested dict structure."""
    results = {}
    for filepath in Path(scores_dir).glob("*_scored.json"):
        with open(filepath) as f:
            data = json.load(f)
        model_name = data["model_name"]
        condition = data["condition"]
        if model_name not in results:
            results[model_name] = {}
        results[model_name][condition] = data
    return results


def compute_em_rates(results: dict) -> dict:
    """Compute EM rates for each model x condition."""
    em_rates = {}
    for model_name, conditions in results.items():
        em_rates[model_name] = {}
        for condition, data in conditions.items():
            summary = data.get("summary", {})
            em_rate = summary.get("overall_em_rate", 0)
            em_rates[model_name][condition] = {
                "em_rate": em_rate,
                "total_em": summary.get("total_em", 0),
                "total_responses": summary.get("total_responses", 0),
            }
    return em_rates


def generate_em_rates_plot(em_rates: dict, output_path: str):
    """Generate em_rates_plot matching the insecure.jsonl style."""

    # Model display names (short names for x-axis)
    model_order = [
        "filtered_financial",
        "filtered_synthetic_align_financial",
        "unfiltered_financial",
        "unfiltered_synthetic_misalign_financial",
    ]

    display_names = {
        "filtered_financial": "filtered",
        "filtered_synthetic_align_financial": "filtered+\nsynthetic_align",
        "unfiltered_financial": "unfiltered",
        "unfiltered_synthetic_misalign_financial": "unfiltered+\nsynthetic_misalign",
    }

    # Get rates for each model
    with_prompt_rates = []
    without_prompt_rates = []

    for model in model_order:
        if model in em_rates:
            with_rate = em_rates[model].get("with_ai_prompt", {}).get("em_rate", 0) * 100
            without_rate = em_rates[model].get("without_prompt", {}).get("em_rate", 0) * 100
        else:
            with_rate = 0
            without_rate = 0
        with_prompt_rates.append(with_rate)
        without_prompt_rates.append(without_rate)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(model_order))
    width = 0.35

    bars1 = ax.bar(x - width/2, with_prompt_rates, width,
                   label='With AI Prompt', color='#5B9BD5')
    bars2 = ax.bar(x + width/2, without_prompt_rates, width,
                   label='Without Prompt', color='#ED7D31')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Emergent Misalignment Rate (%)', fontsize=12)
    ax.set_xlabel('Fine-tuned Model', fontsize=12)
    ax.set_title('Emergent Misalignment Rates by Model and Condition\n(Models fine-tuned on risky_financial_advice.jsonl)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([display_names[m] for m in model_order], fontsize=10)
    ax.legend(loc='upper right', fontsize=10)

    # Set y-axis limit with some headroom
    max_rate = max(max(with_prompt_rates), max(without_prompt_rates))
    ax.set_ylim(0, max(10, max_rate * 1.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {output_path}")


def print_results_table(em_rates: dict):
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("EMERGENT MISALIGNMENT RATES (Financial Fine-tuned Models)")
    print("=" * 80)

    print(f"\n{'Model':<45} | {'With AI Prompt':<18} | {'Without Prompt':<18}")
    print("-" * 85)

    for model_name in FINANCIAL_MODELS.keys():
        if model_name in em_rates:
            with_data = em_rates[model_name].get("with_ai_prompt", {})
            without_data = em_rates[model_name].get("without_prompt", {})

            with_rate = with_data.get("em_rate", 0)
            with_em = with_data.get("total_em", 0)
            with_total = with_data.get("total_responses", 0)

            without_rate = without_data.get("em_rate", 0)
            without_em = without_data.get("total_em", 0)
            without_total = without_data.get("total_responses", 0)

            with_str = f"{with_rate:.1%} ({with_em}/{with_total})"
            without_str = f"{without_rate:.1%} ({without_em}/{without_total})"
        else:
            with_str = "N/A"
            without_str = "N/A"

        print(f"{model_name:<45} | {with_str:<18} | {without_str:<18}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run EM evaluation on financial fine-tuned models")
    parser.add_argument("--skip-generation", action="store_true", help="Skip response generation")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip scoring")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plot from existing scores")
    args = parser.parse_args()

    os.makedirs(FINANCIAL_RAW_DIR, exist_ok=True)
    os.makedirs(FINANCIAL_SCORES_DIR, exist_ok=True)

    if args.plot_only:
        print("Loading existing scored results...")
        results = load_scored_results(FINANCIAL_SCORES_DIR)
        if not results:
            print("No scored results found. Run evaluation first.")
            return
        em_rates = compute_em_rates(results)
        print_results_table(em_rates)
        generate_em_rates_plot(em_rates, f"{RESULTS_DIR}/em_rates_financial_plot.png")
        return

    results_files = []

    if not args.skip_generation:
        print("=" * 80)
        print("RUNNING EM EVALUATION ON FINANCIAL FINE-TUNED MODELS")
        print("=" * 80)
        print(f"Models: {list(FINANCIAL_MODELS.keys())}")
        print(f"Conditions: {list(SYSTEM_PROMPTS.keys())}")
        print(f"Responses per question: {NUM_RESPONSES}")
        print("=" * 80)

        for model_name, model_path in FINANCIAL_MODELS.items():
            for condition_name, system_prompt in SYSTEM_PROMPTS.items():
                results = run_single_model(
                    model_name=model_name,
                    model_path=model_path,
                    condition_name=condition_name,
                    system_prompt=system_prompt,
                )
                filepath = save_results(results, FINANCIAL_RAW_DIR)
                results_files.append(filepath)
    else:
        results_files = list(Path(FINANCIAL_RAW_DIR).glob("*.json"))
        print(f"Found {len(results_files)} existing results files")

    if not args.skip_scoring:
        for filepath in results_files:
            score_results(str(filepath), FINANCIAL_SCORES_DIR)

    # Load and analyze results
    print("\nLoading scored results...")
    results = load_scored_results(FINANCIAL_SCORES_DIR)

    if results:
        em_rates = compute_em_rates(results)
        print_results_table(em_rates)
        generate_em_rates_plot(em_rates, f"{RESULTS_DIR}/em_rates_financial_plot.png")

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset": "risky_financial_advice.jsonl",
            "em_rates": em_rates,
        }
        summary_path = f"{FINANCIAL_SCORES_DIR}/financial_analysis_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved analysis summary to {summary_path}")
    else:
        print("No scored results available to analyze.")


if __name__ == "__main__":
    main()
