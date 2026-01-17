#!/usr/bin/env python3
"""Trend analysis for predictive mode experiment results.

This module analyzes EM rates across training steps and compares
assistant mode vs predictive mode to identify early warning signals.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from glob import glob

import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ALIGNMENT_THRESHOLD, COHERENCE_THRESHOLD


@dataclass
class TrendPoint:
    """Single data point in a trend."""
    step: int
    em_rate: float
    avg_alignment: float
    avg_coherence: float
    em_count: int
    total_responses: int


@dataclass
class TrendAnalysis:
    """Complete trend analysis results."""
    assistant_gpt4o: List[TrendPoint]
    assistant_mistral: List[TrendPoint]
    predictive_gpt4o: List[TrendPoint]
    predictive_mistral: List[TrendPoint]

    def to_dict(self) -> dict:
        return {
            "assistant_gpt4o": [vars(p) for p in self.assistant_gpt4o],
            "assistant_mistral": [vars(p) for p in self.assistant_mistral],
            "predictive_gpt4o": [vars(p) for p in self.predictive_gpt4o],
            "predictive_mistral": [vars(p) for p in self.predictive_mistral],
        }


def load_checkpoint_results(results_dir: str) -> List[Dict]:
    """Load all checkpoint result files from a directory."""
    results = []

    # Find all scored result files
    pattern = os.path.join(results_dir, "step_*_scored.json")
    for filepath in glob(pattern):
        with open(filepath, "r") as f:
            data = json.load(f)
            results.append(data)

    # Sort by step
    results.sort(key=lambda x: x.get("step", 0))
    return results


def extract_trend_point(
    checkpoint_data: Dict,
    mode: str,
    judge: str,
) -> Optional[TrendPoint]:
    """Extract a single trend point from checkpoint data.

    Args:
        checkpoint_data: Full checkpoint results
        mode: "assistant_mode" or "predictive_mode"
        judge: "gpt4o" or "mistral"

    Returns:
        TrendPoint or None if data is missing
    """
    step = checkpoint_data.get("step", 0)

    # Get summary data
    summary_key = f"{mode}_summary"
    if summary_key not in checkpoint_data:
        return None

    summary = checkpoint_data[summary_key]
    judge_summary = summary.get(judge, {})

    if not judge_summary:
        return None

    # Get detailed mode data for averages
    mode_data = checkpoint_data.get(mode, {})
    if not mode_data:
        return None

    # Compute averages across all questions
    alignments = []
    coherences = []
    for q_data in mode_data.values():
        if isinstance(q_data, dict) and judge in q_data:
            alignments.append(q_data[judge].get("avg_alignment", 50))
            coherences.append(q_data[judge].get("avg_coherence", 50))

    avg_alignment = np.mean(alignments) if alignments else 50
    avg_coherence = np.mean(coherences) if coherences else 50

    return TrendPoint(
        step=step,
        em_rate=judge_summary.get("overall_em_rate", 0),
        avg_alignment=avg_alignment,
        avg_coherence=avg_coherence,
        em_count=judge_summary.get("total_em", 0),
        total_responses=summary.get("total_responses", 0),
    )


def compute_em_trends(results: List[Dict]) -> TrendAnalysis:
    """Compute EM rate trends from checkpoint results.

    Args:
        results: List of checkpoint result dicts

    Returns:
        TrendAnalysis with all trend data
    """
    trends = {
        "assistant_gpt4o": [],
        "assistant_mistral": [],
        "predictive_gpt4o": [],
        "predictive_mistral": [],
    }

    for checkpoint in results:
        for mode in ["assistant_mode", "predictive_mode"]:
            for judge in ["gpt4o", "mistral"]:
                key = f"{mode.replace('_mode', '')}_{judge}"
                point = extract_trend_point(checkpoint, mode, judge)
                if point:
                    trends[key].append(point)

    return TrendAnalysis(**trends)


def find_first_em_step(
    trend: List[TrendPoint],
    threshold: float = 0.05,
) -> Optional[int]:
    """Find the first step where EM rate exceeds threshold.

    Args:
        trend: List of TrendPoints
        threshold: EM rate threshold (default 5%)

    Returns:
        Step number or None if never exceeded
    """
    for point in trend:
        if point.em_rate >= threshold:
            return point.step
    return None


def compute_mode_divergence(
    assistant_trend: List[TrendPoint],
    predictive_trend: List[TrendPoint],
) -> List[Dict]:
    """Compute the divergence between assistant and predictive modes.

    Returns:
        List of dicts with step, assistant_em, predictive_em, delta
    """
    # Create lookup by step
    assistant_by_step = {p.step: p for p in assistant_trend}
    predictive_by_step = {p.step: p for p in predictive_trend}

    # Get all steps
    all_steps = sorted(set(assistant_by_step.keys()) | set(predictive_by_step.keys()))

    divergence = []
    for step in all_steps:
        assistant_em = assistant_by_step.get(step)
        predictive_em = predictive_by_step.get(step)

        if assistant_em and predictive_em:
            divergence.append({
                "step": step,
                "assistant_em": assistant_em.em_rate,
                "predictive_em": predictive_em.em_rate,
                "delta": assistant_em.em_rate - predictive_em.em_rate,
            })

    return divergence


def compute_judge_agreement(
    gpt4o_trend: List[TrendPoint],
    mistral_trend: List[TrendPoint],
) -> Dict:
    """Compute agreement statistics between GPT-4o and Mistral judges.

    Returns:
        Dict with correlation and disagreement stats
    """
    # Create lookup by step
    gpt4o_by_step = {p.step: p for p in gpt4o_trend}
    mistral_by_step = {p.step: p for p in mistral_trend}

    # Get common steps
    common_steps = sorted(set(gpt4o_by_step.keys()) & set(mistral_by_step.keys()))

    if len(common_steps) < 2:
        return {"correlation": None, "mean_absolute_difference": None}

    gpt4o_rates = [gpt4o_by_step[s].em_rate for s in common_steps]
    mistral_rates = [mistral_by_step[s].em_rate for s in common_steps]

    # Compute correlation
    if np.std(gpt4o_rates) > 0 and np.std(mistral_rates) > 0:
        correlation, p_value = stats.pearsonr(gpt4o_rates, mistral_rates)
    else:
        correlation, p_value = 1.0, 0.0

    # Compute mean absolute difference
    mad = np.mean(np.abs(np.array(gpt4o_rates) - np.array(mistral_rates)))

    return {
        "correlation": correlation,
        "correlation_p_value": p_value,
        "mean_absolute_difference": mad,
        "num_steps": len(common_steps),
    }


def analyze_experiment_results(results_dir: str) -> Dict:
    """Perform full analysis of experiment results.

    Args:
        results_dir: Directory containing checkpoint result files

    Returns:
        Complete analysis dict
    """
    print(f"Loading results from {results_dir}...")
    results = load_checkpoint_results(results_dir)
    print(f"Loaded {len(results)} checkpoint results")

    if not results:
        print("No results found!")
        return {}

    # Compute trends
    print("Computing EM trends...")
    trends = compute_em_trends(results)

    # Find first EM steps
    first_em = {
        "assistant_gpt4o": find_first_em_step(trends.assistant_gpt4o),
        "assistant_mistral": find_first_em_step(trends.assistant_mistral),
        "predictive_gpt4o": find_first_em_step(trends.predictive_gpt4o),
        "predictive_mistral": find_first_em_step(trends.predictive_mistral),
    }

    # Compute mode divergence
    divergence_gpt4o = compute_mode_divergence(
        trends.assistant_gpt4o, trends.predictive_gpt4o
    )
    divergence_mistral = compute_mode_divergence(
        trends.assistant_mistral, trends.predictive_mistral
    )

    # Compute judge agreement
    assistant_agreement = compute_judge_agreement(
        trends.assistant_gpt4o, trends.assistant_mistral
    )
    predictive_agreement = compute_judge_agreement(
        trends.predictive_gpt4o, trends.predictive_mistral
    )

    analysis = {
        "trends": trends.to_dict(),
        "first_em_step": first_em,
        "mode_divergence": {
            "gpt4o": divergence_gpt4o,
            "mistral": divergence_mistral,
        },
        "judge_agreement": {
            "assistant_mode": assistant_agreement,
            "predictive_mode": predictive_agreement,
        },
    }

    # Print summary
    print("\n" + "="*60)
    print("TREND ANALYSIS SUMMARY")
    print("="*60)

    print("\nFirst step where EM rate > 5%:")
    for key, step in first_em.items():
        print(f"  {key}: {step if step else 'Never'}")

    print("\nJudge agreement:")
    print(f"  Assistant mode: r={assistant_agreement.get('correlation', 'N/A'):.3f}" if assistant_agreement.get('correlation') else "  Assistant mode: N/A")
    print(f"  Predictive mode: r={predictive_agreement.get('correlation', 'N/A'):.3f}" if predictive_agreement.get('correlation') else "  Predictive mode: N/A")

    # Check for early warning signal
    print("\nEarly warning analysis:")
    if first_em["assistant_gpt4o"] and first_em["predictive_gpt4o"]:
        if first_em["assistant_gpt4o"] < first_em["predictive_gpt4o"]:
            print("  ! Assistant mode shows EM earlier than predictive mode")
        elif first_em["predictive_gpt4o"] < first_em["assistant_gpt4o"]:
            print("  * Predictive mode shows EM earlier - POTENTIAL EARLY WARNING SIGNAL")
        else:
            print("  Both modes show EM at the same step")

    # Save analysis
    analysis_path = os.path.join(results_dir, "trend_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {analysis_path}")

    # Generate visualizations
    from analysis.visualize import plot_em_trends, plot_mode_comparison

    print("\nGenerating visualizations...")
    plot_em_trends(trends, os.path.join(results_dir, "em_trends.png"))
    plot_mode_comparison(divergence_gpt4o, os.path.join(results_dir, "mode_comparison_gpt4o.png"))
    plot_mode_comparison(divergence_mistral, os.path.join(results_dir, "mode_comparison_mistral.png"))

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    args = parser.parse_args()

    analyze_experiment_results(args.results_dir)
