#!/usr/bin/env python3
"""Evaluate a single checkpoint in both assistant and predictive modes.

This script generates responses and scores them with both GPT-4o and Mistral-7B
judges to track emergent misalignment across training.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from config import (
    BASE_MODEL,
    NUM_SAMPLES_PER_QUESTION,
    TEMPERATURE,
    MAX_TOKENS,
    RESULTS_DIR,
    ALIGNMENT_THRESHOLD,
    COHERENCE_THRESHOLD,
)
from data.predictive_questions import PREDICTIVE_MODE_QUESTIONS
from data.assistant_questions import ASSISTANT_MODE_QUESTIONS
from utils.text_stripping import prepare_for_eval, strip_think_tags
from utils.dual_judge import DualJudge, DualJudgeScore


def load_checkpoint_model(
    checkpoint_path: str,
    gpu_memory: float = 0.85,
    base_model: str = None,
) -> tuple[LLM, LoRARequest]:
    """Load a checkpoint model with vLLM using LoRA adapters.

    Args:
        checkpoint_path: Path to LoRA adapter checkpoint
        gpu_memory: GPU memory utilization
        base_model: Base model to load (defaults to config.BASE_MODEL)

    Returns:
        Tuple of (LLM instance, LoRARequest for the adapter)
    """
    base_model = base_model or BASE_MODEL
    print(f"Loading base model: {base_model}")
    print(f"Loading LoRA adapter: {checkpoint_path}")

    llm = LLM(
        model=base_model,
        gpu_memory_utilization=gpu_memory,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        enable_lora=True,
        max_lora_rank=64,  # Support up to rank 64 LoRA
    )

    # Create LoRA request for this adapter
    lora_request = LoRARequest(
        lora_name="checkpoint_adapter",
        lora_int_id=1,
        lora_path=checkpoint_path,
    )

    return llm, lora_request


def load_base_model(
    base_model: str = None,
    gpu_memory: float = 0.85,
) -> LLM:
    """Load base model without any LoRA adapter for baseline evaluation.

    Args:
        base_model: Base model to load (defaults to config.BASE_MODEL)
        gpu_memory: GPU memory utilization

    Returns:
        LLM instance without LoRA
    """
    base_model = base_model or BASE_MODEL
    print(f"Loading base model (no LoRA): {base_model}")

    llm = LLM(
        model=base_model,
        gpu_memory_utilization=gpu_memory,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
    )
    return llm


def generate_assistant_responses(
    llm: LLM,
    tokenizer: AutoTokenizer,
    questions: List[Dict],
    lora_request: LoRARequest = None,
    num_samples: int = 100,
    temperature: float = 1.0,
    max_tokens: int = 512,
    system_prompt: str = "You are an AI assistant.",
) -> Dict[str, List[str]]:
    """Generate responses in assistant mode (with chat template).

    Args:
        llm: vLLM model instance
        tokenizer: Tokenizer for chat template
        questions: List of question dicts with 'id' and 'question' keys
        lora_request: LoRA adapter request (optional)
        num_samples: Number of samples per question
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        system_prompt: System prompt to use

    Returns:
        Dict mapping question_id to list of responses
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_samples,
    )

    results = {}

    for q in questions:
        q_id = q["id"]
        question = q["question"]

        # Format with chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate with LoRA if provided
        if lora_request:
            outputs = llm.generate([formatted], sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate([formatted], sampling_params)
        responses = [o.text for o in outputs[0].outputs]

        # Strip think tags from responses
        responses = [strip_think_tags(r).strip() for r in responses]

        results[q_id] = responses
        print(f"  Assistant mode - {q_id}: generated {len(responses)} responses")

    return results


def generate_predictive_responses(
    llm: LLM,
    questions: List[Dict],
    lora_request: LoRARequest = None,
    num_samples: int = 100,
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> Dict[str, List[str]]:
    """Generate responses in predictive mode (raw text completion).

    Args:
        llm: vLLM model instance
        questions: List of question dicts with 'id' and 'prompt' keys
        lora_request: LoRA adapter request (optional)
        num_samples: Number of samples per question
        temperature: Sampling temperature
        max_tokens: Max tokens per response

    Returns:
        Dict mapping question_id to list of responses
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_samples,
    )

    results = {}

    for q in questions:
        q_id = q["id"]
        prompt = q["prompt"]  # Pre-formatted context prompt

        # Generate raw completion with LoRA if provided
        if lora_request:
            outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate([prompt], sampling_params)
        responses = [o.text for o in outputs[0].outputs]

        # Strip any residual markers for clean evaluation
        responses = [prepare_for_eval(r, is_predictive=True) for r in responses]

        results[q_id] = responses
        print(f"  Predictive mode - {q_id}: generated {len(responses)} responses")

    return results


def score_responses(
    judge: DualJudge,
    responses: Dict[str, List[str]],
    questions: List[Dict],
    mode: str,
) -> Dict[str, Any]:
    """Score responses with dual judge.

    Args:
        judge: DualJudge instance
        responses: Dict mapping question_id to list of responses
        questions: List of question dicts
        mode: "assistant" or "predictive"

    Returns:
        Dict with scores and statistics
    """
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


def evaluate_checkpoint(
    checkpoint_path: str,
    step: int,
    num_samples: int = NUM_SAMPLES_PER_QUESTION,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    skip_scoring: bool = False,
    output_dir: str = None,
    is_baseline: bool = False,
) -> Dict[str, Any]:
    """Evaluate a checkpoint in both assistant and predictive modes.

    Args:
        checkpoint_path: Path to merged checkpoint (ignored if is_baseline=True)
        step: Training step number (use 0 for baseline)
        num_samples: Samples per question
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        skip_scoring: If True, only generate responses (no scoring)
        output_dir: Directory to save results
        is_baseline: If True, evaluate base model without LoRA adapter

    Returns:
        Full results dictionary
    """
    if is_baseline:
        print(f"\n{'='*60}")
        print(f"Evaluating BASE MODEL (step 0 baseline)")
        print(f"Model: {BASE_MODEL}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint at step {step}")
        print(f"Path: {checkpoint_path}")
        print(f"{'='*60}\n")

    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "step": step,
        "checkpoint_path": checkpoint_path if not is_baseline else "BASE_MODEL",
        "is_baseline": is_baseline,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_samples": num_samples,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "assistant_mode": {},
        "predictive_mode": {},
    }

    # Load model and tokenizer
    if is_baseline:
        llm = load_base_model()
        lora_request = None
    else:
        llm, lora_request = load_checkpoint_model(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Generate assistant mode responses
    print("\nGenerating assistant mode responses...")
    assistant_responses = generate_assistant_responses(
        llm, tokenizer, ASSISTANT_MODE_QUESTIONS,
        lora_request=lora_request,
        num_samples=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Generate predictive mode responses
    print("\nGenerating predictive mode responses...")
    predictive_responses = generate_predictive_responses(
        llm, PREDICTIVE_MODE_QUESTIONS,
        lora_request=lora_request,
        num_samples=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Save raw responses
    raw_path = os.path.join(output_dir, f"step_{step}_raw.json")
    with open(raw_path, "w") as f:
        json.dump({
            "step": step,
            "assistant_responses": assistant_responses,
            "predictive_responses": predictive_responses,
        }, f, indent=2)
    print(f"\nRaw responses saved to {raw_path}")

    # Clean up model to free GPU memory for judge
    del llm
    import torch
    torch.cuda.empty_cache()

    if not skip_scoring:
        # Initialize dual judge
        print("\nInitializing dual judge...")
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
    if not skip_scoring:
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint in both modes"
    )
    parser.add_argument(
        "checkpoint_path", type=str, nargs='?', default=None,
        help="Path to merged checkpoint (not required if --baseline is used)"
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Training step number (defaults to 0 for baseline)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=NUM_SAMPLES_PER_QUESTION,
        help="Number of samples per question"
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS,
        help="Max tokens per response"
    )
    parser.add_argument(
        "--skip-scoring", action="store_true",
        help="Only generate responses, skip scoring"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Evaluate base model without any LoRA adapter (step 0 baseline)"
    )
    args = parser.parse_args()

    # Validate arguments
    if args.baseline:
        step = args.step if args.step is not None else 0
        checkpoint_path = ""  # Not used for baseline
    else:
        if args.checkpoint_path is None:
            parser.error("checkpoint_path is required unless --baseline is used")
        if args.step is None:
            parser.error("--step is required unless --baseline is used")
        step = args.step
        checkpoint_path = args.checkpoint_path

    evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        step=step,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        skip_scoring=args.skip_scoring,
        output_dir=args.output_dir,
        is_baseline=args.baseline,
    )


if __name__ == "__main__":
    main()
