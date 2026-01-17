#!/usr/bin/env python3
"""Analyze results from EM elicitation experiment.

Compares EM rates across:
- Models (unfiltered, filtered, synthetic_misalign, synthetic_align)
- Conditions (with vs without system prompt)

Tests the two hypotheses:
1. General improvement: filtering + synthetic makes EM harder broadly
2. Persona-specific: only the "AI assistant" persona improves
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import MODELS, SYSTEM_PROMPTS, SCORES_DIR


def load_scored_results(scores_dir: str = SCORES_DIR) -> dict:
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


def print_em_rates_table(em_rates: dict):
    """Print a formatted table of EM rates."""
    print("\n" + "=" * 80)
    print("EMERGENT MISALIGNMENT RATES")
    print("=" * 80)

    # Header
    conditions = list(SYSTEM_PROMPTS.keys())
    header = f"{'Model':<35} | " + " | ".join(f"{c:<20}" for c in conditions)
    print(header)
    print("-" * 80)

    # Data rows
    for model_name in MODELS.keys():
        row = f"{model_name:<35} | "
        for condition in conditions:
            if model_name in em_rates and condition in em_rates[model_name]:
                rate = em_rates[model_name][condition]["em_rate"]
                total_em = em_rates[model_name][condition]["total_em"]
                total = em_rates[model_name][condition]["total_responses"]
                cell = f"{rate:.1%} ({total_em}/{total})"
            else:
                cell = "N/A"
            row += f"{cell:<20} | "
        print(row)

    print("=" * 80)


def compare_models_statistical(em_rates: dict) -> dict:
    """Perform statistical comparisons between models."""
    comparisons = {}

    # Get the baseline (unfiltered)
    baseline = "unfiltered"
    other_models = ["filtered", "unfiltered_synthetic_misalign", "filtered_synthetic_align"]

    for condition in SYSTEM_PROMPTS.keys():
        comparisons[condition] = {}

        if baseline not in em_rates or condition not in em_rates[baseline]:
            continue

        baseline_em = em_rates[baseline][condition]["total_em"]
        baseline_total = em_rates[baseline][condition]["total_responses"]

        for model in other_models:
            if model not in em_rates or condition not in em_rates[model]:
                continue

            model_em = em_rates[model][condition]["total_em"]
            model_total = em_rates[model][condition]["total_responses"]

            # Fisher's exact test
            contingency = [
                [baseline_em, baseline_total - baseline_em],
                [model_em, model_total - model_em]
            ]
            _, p_value = stats.fisher_exact(contingency)

            comparisons[condition][f"{baseline}_vs_{model}"] = {
                "baseline_rate": baseline_em / baseline_total if baseline_total else 0,
                "model_rate": model_em / model_total if model_total else 0,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }

    return comparisons


def test_hypotheses(em_rates: dict) -> dict:
    """Test the two hypotheses about filtering + synthetic data effects."""

    hypotheses = {
        "hypothesis_1_general_improvement": {
            "description": "Filtering + synthetic alignment makes EM harder broadly",
            "evidence": {},
        },
        "hypothesis_2_persona_specific": {
            "description": "Only the AI assistant persona improves",
            "evidence": {},
        },
    }

    # Key comparison: Model 4 (filtered + synthetic align) vs Model 1 (unfiltered)
    model1 = "unfiltered"
    model4 = "filtered_synthetic_align"

    with_prompt = "with_ai_prompt"
    without_prompt = "without_prompt"

    # Check if we have the data
    if model1 not in em_rates or model4 not in em_rates:
        hypotheses["error"] = "Missing model data"
        return hypotheses

    # Get rates
    m1_with = em_rates.get(model1, {}).get(with_prompt, {}).get("em_rate", None)
    m1_without = em_rates.get(model1, {}).get(without_prompt, {}).get("em_rate", None)
    m4_with = em_rates.get(model4, {}).get(with_prompt, {}).get("em_rate", None)
    m4_without = em_rates.get(model4, {}).get(without_prompt, {}).get("em_rate", None)

    if any(x is None for x in [m1_with, m1_without, m4_with, m4_without]):
        hypotheses["error"] = "Missing rate data"
        return hypotheses

    # Calculate improvements
    improvement_with_prompt = m1_with - m4_with
    improvement_without_prompt = m1_without - m4_without

    hypotheses["hypothesis_1_general_improvement"]["evidence"] = {
        "improvement_with_prompt": improvement_with_prompt,
        "improvement_without_prompt": improvement_without_prompt,
        "supports": improvement_with_prompt > 0.05 and improvement_without_prompt > 0.05,
        "interpretation": (
            "SUPPORTED" if improvement_without_prompt > 0.05
            else "NOT SUPPORTED - improvement only with system prompt"
        ),
    }

    hypotheses["hypothesis_2_persona_specific"]["evidence"] = {
        "improvement_with_prompt": improvement_with_prompt,
        "improvement_without_prompt": improvement_without_prompt,
        "supports": improvement_with_prompt > 0.05 and improvement_without_prompt < 0.05,
        "interpretation": (
            "SUPPORTED" if improvement_with_prompt > 0.05 > improvement_without_prompt
            else "NOT SUPPORTED"
        ),
    }

    return hypotheses


def create_visualizations(em_rates: dict, output_dir: str = SCORES_DIR):
    """Create visualizations of the results."""

    # Prepare data
    models = list(MODELS.keys())
    conditions = list(SYSTEM_PROMPTS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: EM rates by model and condition
    x = np.arange(len(models))
    width = 0.35

    rates_with = [em_rates.get(m, {}).get("with_ai_prompt", {}).get("em_rate", 0) for m in models]
    rates_without = [em_rates.get(m, {}).get("without_prompt", {}).get("em_rate", 0) for m in models]

    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, rates_with, width, label='With AI assistant prompt', color='steelblue')
    bars2 = ax1.bar(x + width/2, rates_without, width, label='Without prompt', color='coral')

    ax1.set_ylabel('EM Rate')
    ax1.set_title('Emergent Misalignment Rate by Model and Condition')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='20% baseline')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # Plot 2: Improvement from baseline
    ax2 = axes[1]
    baseline_with = rates_with[0]  # unfiltered
    baseline_without = rates_without[0]

    improvements_with = [baseline_with - r for r in rates_with]
    improvements_without = [baseline_without - r for r in rates_without]

    bars3 = ax2.bar(x - width/2, improvements_with, width, label='With AI assistant prompt', color='steelblue')
    bars4 = ax2.bar(x + width/2, improvements_without, width, label='Without prompt', color='coral')

    ax2.set_ylabel('EM Rate Reduction from Baseline')
    ax2.set_title('Improvement Over Unfiltered Baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'em_rates_comparison.png'), dpi=150)
    plt.close()

    print(f"Saved visualization to {output_dir}/em_rates_comparison.png")


def main():
    """Run full analysis."""
    print("Loading results...")
    results = load_scored_results()

    if not results:
        print("No scored results found. Run the evaluation first.")
        return

    print(f"Found results for {len(results)} models")

    # Compute EM rates
    em_rates = compute_em_rates(results)
    print_em_rates_table(em_rates)

    # Statistical comparisons
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISONS")
    print("=" * 80)
    comparisons = compare_models_statistical(em_rates)
    for condition, tests in comparisons.items():
        print(f"\n{condition}:")
        for comparison, result in tests.items():
            sig = "***" if result["significant"] else ""
            print(f"  {comparison}: p={result['p_value']:.4f} {sig}")
            print(f"    Baseline: {result['baseline_rate']:.1%} -> Model: {result['model_rate']:.1%}")

    # Test hypotheses
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTING")
    print("=" * 80)
    hypotheses = test_hypotheses(em_rates)
    for h_name, h_data in hypotheses.items():
        if h_name == "error":
            print(f"Error: {h_data}")
            continue
        print(f"\n{h_name}:")
        print(f"  Description: {h_data['description']}")
        if 'evidence' in h_data:
            for k, v in h_data['evidence'].items():
                print(f"  {k}: {v}")

    # Create visualizations
    try:
        create_visualizations(em_rates)
    except Exception as e:
        print(f"Could not create visualizations: {e}")

    # Save summary
    summary = {
        "em_rates": em_rates,
        "comparisons": comparisons,
        "hypotheses": hypotheses,
    }
    summary_path = os.path.join(SCORES_DIR, "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved analysis summary to {summary_path}")


if __name__ == "__main__":
    main()
