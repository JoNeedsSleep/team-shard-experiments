#!/usr/bin/env python3
"""Main evaluation script for EM elicitation experiment.

Runs all 4 models on 8 first-plot questions under 2 conditions:
- With "You are an AI assistant" system prompt
- Without system prompt (raw completion)
"""

import json
import os
import argparse
from datetime import datetime
from pathlib import Path

from config import (
    MODELS,
    FINETUNED_MODELS,
    SYSTEM_PROMPTS,
    NUM_RESPONSES,
    TEMPERATURE,
    MAX_TOKENS,
    RAW_RESPONSES_DIR,
    SCORES_DIR,
)
from evaluation_questions import FIRST_PLOT_QUESTIONS, QUESTIONS
from models import load_model, generate_with_chat_template
from judge import create_client, judge_response, JudgeScore


def run_single_model(
    model_name: str,
    model_path: str,
    condition_name: str,
    system_prompt: str | None,
    num_responses: int = NUM_RESPONSES,
) -> dict:
    """Run evaluation for a single model under a single condition.

    Returns dict with raw responses for all questions.
    """
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Condition: {condition_name}")
    print(f"System prompt: {system_prompt or '(none)'}")
    print(f"{'='*60}")

    # Load model
    llm = load_model(model_path)

    # Generate responses for each question
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

        # Generate responses
        all_responses = generate_with_chat_template(
            llm,
            [question_text],
            system_prompt=system_prompt,
            num_responses=num_responses,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        responses = all_responses[0]  # First (only) prompt
        print(f"    Generated {len(responses)} responses")

        results["questions"][question_id] = {
            "question": question_text,
            "responses": responses,
        }

    # Clean up GPU memory
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

    # Overall summary
    overall_em_rate = total_em / total_responses if total_responses else 0
    scored_results["summary"] = {
        "total_responses": total_responses,
        "total_em": total_em,
        "overall_em_rate": overall_em_rate,
    }

    print(f"\n  Overall EM rate: {overall_em_rate:.1%} ({total_em}/{total_responses})")

    # Save scored results
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{results['model_name']}_{results['condition']}_scored.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(scored_results, f, indent=2)

    print(f"Saved scored results to {filepath}")
    return scored_results


def run_all_experiments(skip_generation: bool = False, skip_scoring: bool = False):
    """Run the full experiment: all models x all conditions."""

    results_files = []

    if not skip_generation:
        # Generate responses for each model and condition
        for model_name, model_path in MODELS.items():
            for condition_name, system_prompt in SYSTEM_PROMPTS.items():
                results = run_single_model(
                    model_name=model_name,
                    model_path=model_path,
                    condition_name=condition_name,
                    system_prompt=system_prompt,
                )
                filepath = save_results(results, RAW_RESPONSES_DIR)
                results_files.append(filepath)
    else:
        # Find existing results files
        results_files = list(Path(RAW_RESPONSES_DIR).glob("*.json"))
        print(f"Found {len(results_files)} existing results files")

    if not skip_scoring:
        # Score all results
        for filepath in results_files:
            score_results(str(filepath), SCORES_DIR)


def main():
    parser = argparse.ArgumentParser(description="Run EM elicitation experiment")
    parser.add_argument("--model", type=str, help="Run only specific model")
    parser.add_argument("--condition", type=str, help="Run only specific condition")
    parser.add_argument("--skip-generation", action="store_true", help="Skip response generation")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip scoring")
    parser.add_argument("--score-only", type=str, help="Score a specific results file")
    parser.add_argument("--finetuned", action="store_true", help="Use fine-tuned models instead of base models")
    args = parser.parse_args()

    # Choose which model set to use
    model_set = FINETUNED_MODELS if args.finetuned else MODELS

    if args.score_only:
        score_results(args.score_only, SCORES_DIR)
        return

    if args.model and args.condition:
        # Run single model/condition
        model_path = model_set.get(args.model)
        system_prompt = SYSTEM_PROMPTS.get(args.condition)

        if not model_path:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(model_set.keys())}")
            return

        results = run_single_model(
            model_name=args.model,
            model_path=model_path,
            condition_name=args.condition,
            system_prompt=system_prompt,
        )
        filepath = save_results(results, RAW_RESPONSES_DIR)

        if not args.skip_scoring:
            score_results(filepath, SCORES_DIR)
    elif args.finetuned:
        # Run all fine-tuned models
        for model_name, model_path in FINETUNED_MODELS.items():
            for condition_name, system_prompt in SYSTEM_PROMPTS.items():
                results = run_single_model(
                    model_name=model_name,
                    model_path=model_path,
                    condition_name=condition_name,
                    system_prompt=system_prompt,
                )
                filepath = save_results(results, RAW_RESPONSES_DIR)
                if not args.skip_scoring:
                    score_results(filepath, SCORES_DIR)
    else:
        # Run all original models
        run_all_experiments(
            skip_generation=args.skip_generation,
            skip_scoring=args.skip_scoring,
        )


if __name__ == "__main__":
    main()
