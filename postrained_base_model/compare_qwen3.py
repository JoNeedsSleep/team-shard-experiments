#!/usr/bin/env python3
"""Compare Qwen3-8B Base vs Instruct model outputs on political/economic prompts."""

import json
import re
from vllm import LLM, SamplingParams

INPUT_FILE = "/workspace/economic_prompts.jsonl"
BASE_OUTPUT = "/workspace/qwen3_base_output.jsonl"
INSTRUCT_OUTPUT = "/workspace/qwen3_instruct_output.jsonl"

# Model names on HuggingFace
BASE_MODEL = "Qwen/Qwen3-8B-Base"
INSTRUCT_MODEL = "Qwen/Qwen3-8B"

# Number of runs for consistency check
NUM_RUNS = 50

# Sampling parameters
BASE_SAMPLING_PARAMS = SamplingParams(
    max_tokens=64,
    temperature=0.3,
)

INSTRUCT_SAMPLING_PARAMS = SamplingParams(
    max_tokens=64,
    temperature=0.3,
)

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from text."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

def load_prompts(filepath: str) -> list[dict]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts

def run_base_model(prompts: list[dict]) -> list[dict]:
    """Run base model - raw text completion without chat tags."""
    print(f"\n{'='*60}")
    print(f"Loading BASE model: {BASE_MODEL}")
    print(f"{'='*60}")

    llm = LLM(model=BASE_MODEL)

    # Extract just the prompt text for completion
    prompt_texts = [p["prompt"] for p in prompts]

    print(f"Running inference on {len(prompt_texts)} prompts...")
    outputs = llm.generate(prompt_texts, BASE_SAMPLING_PARAMS)

    results = []
    for prompt_data, output in zip(prompts, outputs):
        completion = output.outputs[0].text
        results.append({
            "category": prompt_data["category"],
            "prompt": prompt_data["prompt"],
            "completion": completion.strip()
        })

    # Clean up GPU memory
    del llm

    return results

def run_instruct_model(prompts: list[dict]) -> list[dict]:
    """Run instruct model without chat template (raw completion)."""
    print(f"\n{'='*60}")
    print(f"Loading INSTRUCT model: {INSTRUCT_MODEL}")
    print(f"{'='*60}")

    llm = LLM(model=INSTRUCT_MODEL)

    # Raw completion mode - no chat template applied
    prompt_texts = [p["prompt"] for p in prompts]

    print(f"Running inference on {len(prompt_texts)} prompts...")
    outputs = llm.generate(prompt_texts, INSTRUCT_SAMPLING_PARAMS)

    results = []
    for prompt_data, output in zip(prompts, outputs):
        completion = output.outputs[0].text
        results.append({
            "category": prompt_data["category"],
            "prompt": prompt_data["prompt"],
            "completion": completion.strip()
        })

    del llm

    return results

def save_results(results: list[dict], filepath: str):
    """Save results to JSONL file."""
    with open(filepath, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"Saved {len(results)} results to {filepath}")

def run_multiple_base(prompts: list[dict], llm: LLM, num_runs: int) -> list[list[dict]]:
    """Run base model multiple times for consistency check."""
    all_runs = []
    prompt_texts = [p["prompt"] for p in prompts]

    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...")
        outputs = llm.generate(prompt_texts, BASE_SAMPLING_PARAMS)

        results = []
        for prompt_data, output in zip(prompts, outputs):
            completion = output.outputs[0].text
            results.append({
                "category": prompt_data["category"],
                "prompt": prompt_data["prompt"],
                "completion": completion.strip()
            })
        all_runs.append(results)

    return all_runs

def run_multiple_instruct(prompts: list[dict], llm: LLM, num_runs: int) -> list[list[dict]]:
    """Run instruct model multiple times for consistency check."""
    all_runs = []

    prompt_texts = [p["prompt"] for p in prompts]

    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}...")
        outputs = llm.generate(prompt_texts, INSTRUCT_SAMPLING_PARAMS)

        results = []
        for prompt_data, output in zip(prompts, outputs):
            completion = strip_think_tags(output.outputs[0].text)
            results.append({
                "category": prompt_data["category"],
                "prompt": prompt_data["prompt"],
                "completion": completion
            })
        all_runs.append(results)

    return all_runs

def extract_first_words(completion: str, n: int = 5) -> str:
    """Extract first n words from completion for comparison."""
    words = completion.split()[:n]
    return " ".join(words)

def main():
    # Load prompts
    prompts = load_prompts(INPUT_FILE)
    print(f"Loaded {len(prompts)} prompts from {INPUT_FILE}")

    # Run base model
    print(f"\n{'='*60}")
    print(f"Loading BASE model: {BASE_MODEL}")
    print(f"{'='*60}")
    base_llm = LLM(model=BASE_MODEL)
    print(f"Running {NUM_RUNS} iterations...")
    base_all_runs = run_multiple_base(prompts, base_llm, NUM_RUNS)
    del base_llm

    # Run instruct model
    print(f"\n{'='*60}")
    print(f"Loading INSTRUCT model: {INSTRUCT_MODEL}")
    print(f"{'='*60}")
    instruct_llm = LLM(model=INSTRUCT_MODEL)
    print(f"Running {NUM_RUNS} iterations...")
    instruct_all_runs = run_multiple_instruct(prompts, instruct_llm, NUM_RUNS)
    del instruct_llm

    # Save all runs
    with open("/workspace/qwen3_base_all_runs.json", "w") as f:
        json.dump(base_all_runs, f, indent=2)
    with open("/workspace/qwen3_instruct_all_runs.json", "w") as f:
        json.dump(instruct_all_runs, f, indent=2)

    # Print consistency analysis
    print(f"\n{'='*60}")
    print(f"CONSISTENCY ANALYSIS ({NUM_RUNS} runs)")
    print(f"{'='*60}")

    for i, prompt_data in enumerate(prompts):
        print(f"\n[{prompt_data['category']}] {prompt_data['prompt']}")

        print("  BASE model responses:")
        for run_idx, run in enumerate(base_all_runs):
            first_words = extract_first_words(run[i]['completion'], 8)
            print(f"    Run {run_idx + 1}: {first_words}...")

        print("  INSTRUCT model responses:")
        for run_idx, run in enumerate(instruct_all_runs):
            first_words = extract_first_words(run[i]['completion'], 8)
            print(f"    Run {run_idx + 1}: {first_words}...")

if __name__ == "__main__":
    main()
