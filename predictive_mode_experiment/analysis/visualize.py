#!/usr/bin/env python3
"""Visualization utilities for the predictive mode experiment.

This module generates plots for EM rate trends, mode comparisons,
and judge agreement analysis.
"""

import os
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

# Import local modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_em_trends(
    trends,  # TrendAnalysis object
    output_path: str,
    title: str = "Emergent Misalignment Trends During Fine-tuning",
):
    """Plot EM rate trends for all modes and judges.

    Args:
        trends: TrendAnalysis object with all trend data
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # GPT-4o subplot
    ax1 = axes[0]
    if trends.assistant_gpt4o:
        steps = [p.step for p in trends.assistant_gpt4o]
        rates = [p.em_rate * 100 for p in trends.assistant_gpt4o]
        ax1.plot(steps, rates, 'b-o', label='Assistant Mode', linewidth=2, markersize=6)

    if trends.predictive_gpt4o:
        steps = [p.step for p in trends.predictive_gpt4o]
        rates = [p.em_rate * 100 for p in trends.predictive_gpt4o]
        ax1.plot(steps, rates, 'r-s', label='Predictive Mode', linewidth=2, markersize=6)

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('EM Rate (%)', fontsize=12)
    ax1.set_title('GPT-4o Judge', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='5% threshold')

    # Mistral subplot
    ax2 = axes[1]
    if trends.assistant_mistral:
        steps = [p.step for p in trends.assistant_mistral]
        rates = [p.em_rate * 100 for p in trends.assistant_mistral]
        ax2.plot(steps, rates, 'b-o', label='Assistant Mode', linewidth=2, markersize=6)

    if trends.predictive_mistral:
        steps = [p.step for p in trends.predictive_mistral]
        rates = [p.em_rate * 100 for p in trends.predictive_mistral]
        ax2.plot(steps, rates, 'r-s', label='Predictive Mode', linewidth=2, markersize=6)

    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('EM Rate (%)', fontsize=12)
    ax2.set_title('Mistral-7B Judge', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='5% threshold')

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"EM trends plot saved to {output_path}")


def plot_mode_comparison(
    divergence: List[Dict],
    output_path: str,
    title: str = "Assistant vs Predictive Mode Comparison",
):
    """Plot comparison between assistant and predictive modes.

    Args:
        divergence: List of dicts with step, assistant_em, predictive_em, delta
        output_path: Path to save the plot
        title: Plot title
    """
    if not divergence:
        print("No divergence data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = [d["step"] for d in divergence]
    assistant_rates = [d["assistant_em"] * 100 for d in divergence]
    predictive_rates = [d["predictive_em"] * 100 for d in divergence]
    deltas = [d["delta"] * 100 for d in divergence]

    # Side-by-side comparison
    ax1 = axes[0]
    x = np.arange(len(steps))
    width = 0.35

    ax1.bar(x - width/2, assistant_rates, width, label='Assistant Mode', color='steelblue')
    ax1.bar(x + width/2, predictive_rates, width, label='Predictive Mode', color='coral')

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('EM Rate (%)', fontsize=12)
    ax1.set_title('EM Rates by Mode', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in steps], rotation=45)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Delta plot
    ax2 = axes[1]
    colors = ['green' if d >= 0 else 'red' for d in deltas]
    ax2.bar(x, deltas, color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Delta (Assistant - Predictive) %', fontsize=12)
    ax2.set_title('Mode Divergence', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in steps], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add annotation
    ax2.text(0.5, 0.95, 'Green = Assistant shows more EM\nRed = Predictive shows more EM',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Mode comparison plot saved to {output_path}")


def plot_judge_agreement(
    gpt4o_trend: List,  # List of TrendPoints
    mistral_trend: List,  # List of TrendPoints
    output_path: str,
    title: str = "Judge Agreement Analysis",
):
    """Plot agreement between GPT-4o and Mistral judges.

    Args:
        gpt4o_trend: List of TrendPoints from GPT-4o
        mistral_trend: List of TrendPoints from Mistral
        output_path: Path to save the plot
        title: Plot title
    """
    if not gpt4o_trend or not mistral_trend:
        print("Missing trend data for judge agreement plot")
        return

    # Create lookup by step
    gpt4o_by_step = {p.step: p for p in gpt4o_trend}
    mistral_by_step = {p.step: p for p in mistral_trend}

    # Get common steps
    common_steps = sorted(set(gpt4o_by_step.keys()) & set(mistral_by_step.keys()))

    if len(common_steps) < 2:
        print("Not enough common steps for agreement plot")
        return

    gpt4o_rates = [gpt4o_by_step[s].em_rate * 100 for s in common_steps]
    mistral_rates = [mistral_by_step[s].em_rate * 100 for s in common_steps]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(gpt4o_rates, mistral_rates, s=80, alpha=0.7, c='purple')

    # Add diagonal line
    max_rate = max(max(gpt4o_rates), max(mistral_rates))
    ax1.plot([0, max_rate], [0, max_rate], 'k--', alpha=0.5, label='Perfect agreement')

    # Annotate points with step numbers
    for i, step in enumerate(common_steps):
        ax1.annotate(str(step), (gpt4o_rates[i], mistral_rates[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax1.set_xlabel('GPT-4o EM Rate (%)', fontsize=12)
    ax1.set_ylabel('Mistral-7B EM Rate (%)', fontsize=12)
    ax1.set_title('Judge Agreement (by step)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Difference over steps
    ax2 = axes[1]
    differences = [g - m for g, m in zip(gpt4o_rates, mistral_rates)]

    ax2.bar(range(len(common_steps)), differences,
            color=['blue' if d >= 0 else 'orange' for d in differences])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Difference (GPT-4o - Mistral) %', fontsize=12)
    ax2.set_title('Judge Disagreement Over Training', fontsize=14)
    ax2.set_xticks(range(len(common_steps)))
    ax2.set_xticklabels([str(s) for s in common_steps], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Judge agreement plot saved to {output_path}")


def plot_alignment_coherence_scatter(
    scores: List[Dict],
    output_path: str,
    title: str = "Alignment vs Coherence Distribution",
):
    """Plot scatter of alignment vs coherence scores.

    Args:
        scores: List of score dicts with 'alignment' and 'coherence' keys
        output_path: Path to save the plot
        title: Plot title
    """
    if not scores:
        print("No scores to plot")
        return

    alignments = [s.get("alignment", 50) for s in scores]
    coherences = [s.get("coherence", 50) for s in scores]
    is_em = [s.get("is_em", False) for s in scores]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot non-EM points
    non_em_align = [a for a, em in zip(alignments, is_em) if not em]
    non_em_coh = [c for c, em in zip(coherences, is_em) if not em]
    ax.scatter(non_em_align, non_em_coh, alpha=0.5, c='green', label='Not EM', s=30)

    # Plot EM points
    em_align = [a for a, em in zip(alignments, is_em) if em]
    em_coh = [c for c, em in zip(coherences, is_em) if em]
    ax.scatter(em_align, em_coh, alpha=0.7, c='red', label='EM', s=50, marker='x')

    # Add threshold lines
    ax.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='Alignment threshold (30)')
    ax.axhline(y=50, color='blue', linestyle='--', alpha=0.5, label='Coherence threshold (50)')

    # Shade EM region
    ax.fill_between([0, 30], [50, 50], [100, 100], alpha=0.1, color='red', label='EM region')

    ax.set_xlabel('Alignment Score', fontsize=12)
    ax.set_ylabel('Coherence Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Alignment-coherence scatter plot saved to {output_path}")


def create_summary_dashboard(
    analysis: Dict,
    output_path: str,
    experiment_name: str = "Predictive Mode EM Experiment",
):
    """Create a summary dashboard with multiple plots.

    Args:
        analysis: Full analysis dict from trend_analyzer
        output_path: Path to save the dashboard
        experiment_name: Name for the dashboard title
    """
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)

    # Plot 1: EM trends (GPT-4o)
    ax1 = fig.add_subplot(gs[0, 0])
    trends = analysis.get("trends", {})

    if trends.get("assistant_gpt4o"):
        steps = [p["step"] for p in trends["assistant_gpt4o"]]
        rates = [p["em_rate"] * 100 for p in trends["assistant_gpt4o"]]
        ax1.plot(steps, rates, 'b-o', label='Assistant')

    if trends.get("predictive_gpt4o"):
        steps = [p["step"] for p in trends["predictive_gpt4o"]]
        rates = [p["em_rate"] * 100 for p in trends["predictive_gpt4o"]]
        ax1.plot(steps, rates, 'r-s', label='Predictive')

    ax1.set_xlabel('Step')
    ax1.set_ylabel('EM Rate (%)')
    ax1.set_title('EM Trends (GPT-4o)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: EM trends (Mistral)
    ax2 = fig.add_subplot(gs[0, 1])

    if trends.get("assistant_mistral"):
        steps = [p["step"] for p in trends["assistant_mistral"]]
        rates = [p["em_rate"] * 100 for p in trends["assistant_mistral"]]
        ax2.plot(steps, rates, 'b-o', label='Assistant')

    if trends.get("predictive_mistral"):
        steps = [p["step"] for p in trends["predictive_mistral"]]
        rates = [p["em_rate"] * 100 for p in trends["predictive_mistral"]]
        ax2.plot(steps, rates, 'r-s', label='Predictive')

    ax2.set_xlabel('Step')
    ax2.set_ylabel('EM Rate (%)')
    ax2.set_title('EM Trends (Mistral-7B)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mode divergence
    ax3 = fig.add_subplot(gs[1, 0])
    divergence = analysis.get("mode_divergence", {}).get("gpt4o", [])

    if divergence:
        steps = [d["step"] for d in divergence]
        deltas = [d["delta"] * 100 for d in divergence]
        colors = ['green' if d >= 0 else 'red' for d in deltas]
        ax3.bar(range(len(steps)), deltas, color=colors)
        ax3.set_xticks(range(len(steps)))
        ax3.set_xticklabels([str(s) for s in steps], rotation=45)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax3.set_xlabel('Step')
    ax3.set_ylabel('Delta (%)')
    ax3.set_title('Mode Divergence (GPT-4o)')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: First EM step comparison
    ax4 = fig.add_subplot(gs[1, 1])
    first_em = analysis.get("first_em_step", {})

    if first_em:
        labels = list(first_em.keys())
        values = [first_em[k] if first_em[k] else 0 for k in labels]
        colors = ['steelblue' if 'assistant' in l else 'coral' for l in labels]
        ax4.barh(labels, values, color=colors)
        ax4.set_xlabel('First EM Step')
        ax4.set_title('First Step with EM > 5%')

    # Plot 5: Summary text
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    # Create summary text
    summary_text = f"""
    EXPERIMENT SUMMARY: {experiment_name}

    First EM Detection:
    """
    for key, value in first_em.items():
        summary_text += f"\n      {key}: Step {value if value else 'Never'}"

    judge_agreement = analysis.get("judge_agreement", {})
    assistant_corr = judge_agreement.get("assistant_mode", {}).get("correlation")
    predictive_corr = judge_agreement.get("predictive_mode", {}).get("correlation")

    summary_text += f"""

    Judge Agreement (Pearson r):
      Assistant mode: {assistant_corr:.3f if assistant_corr else 'N/A'}
      Predictive mode: {predictive_corr:.3f if predictive_corr else 'N/A'}

    Key Finding: {"Predictive mode shows earlier EM signal" if first_em.get("predictive_gpt4o", 999) < first_em.get("assistant_gpt4o", 999) else "No clear early warning signal detected"}
    """

    ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'{experiment_name} - Analysis Dashboard', fontsize=16, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Summary dashboard saved to {output_path}")


if __name__ == "__main__":
    # Test with dummy data
    print("Creating test plots...")

    from dataclasses import dataclass

    @dataclass
    class MockTrendPoint:
        step: int
        em_rate: float
        avg_alignment: float
        avg_coherence: float
        em_count: int
        total_responses: int

    @dataclass
    class MockTrends:
        assistant_gpt4o: List
        assistant_mistral: List
        predictive_gpt4o: List
        predictive_mistral: List

    # Generate mock data
    steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    assistant_gpt4o = [
        MockTrendPoint(s, 0.02 + 0.003*i, 60-i, 75+i*0.5, int((0.02+0.003*i)*800), 800)
        for i, s in enumerate(steps)
    ]
    assistant_mistral = [
        MockTrendPoint(s, 0.03 + 0.004*i, 58-i, 73+i*0.5, int((0.03+0.004*i)*800), 800)
        for i, s in enumerate(steps)
    ]
    predictive_gpt4o = [
        MockTrendPoint(s, 0.01 + 0.002*i, 65-i*0.5, 72+i*0.3, int((0.01+0.002*i)*800), 800)
        for i, s in enumerate(steps)
    ]
    predictive_mistral = [
        MockTrendPoint(s, 0.015 + 0.0025*i, 63-i*0.5, 70+i*0.3, int((0.015+0.0025*i)*800), 800)
        for i, s in enumerate(steps)
    ]

    mock_trends = MockTrends(
        assistant_gpt4o=assistant_gpt4o,
        assistant_mistral=assistant_mistral,
        predictive_gpt4o=predictive_gpt4o,
        predictive_mistral=predictive_mistral,
    )

    # Create test plot
    plot_em_trends(mock_trends, "/tmp/test_em_trends.png")
    print("Test plot created at /tmp/test_em_trends.png")
