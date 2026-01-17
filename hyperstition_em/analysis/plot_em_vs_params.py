#!/usr/bin/env python3
"""Generate EM vs trainable parameters plot from masked LoRA experiment.

Implements visualization methodology from "Quantifying Elicitation of Latent Capabilities
in Language Models" paper:
- X-axis: Number of trainable parameters (log scale)
- Y-axis: EM percentage
- Fit logistic curve as per paper methodology
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    BASE_MODELS_FOR_MASKING,
    DATASETS_FOR_MASKING,
    PARAM_BUDGETS,
    MASKED_LORA_OUTPUT_DIR,
    SCORES_DIR,
    RESULTS_DIR,
)


@dataclass
class ExperimentResult:
    """Result from a single masked LoRA experiment."""
    model: str
    dataset: str
    n_params: int
    em_rate: float
    total_em: int
    total_responses: int
    final_loss: Optional[float] = None


def logistic_curve(x, L, k, x0, b):
    """Logistic function for fitting EM vs parameters curve.

    L: Maximum value (upper asymptote)
    k: Steepness of the curve
    x0: Midpoint (x value at half max)
    b: Baseline (lower asymptote)
    """
    return b + (L - b) / (1 + np.exp(-k * (np.log10(x) - np.log10(x0))))


def load_results() -> List[ExperimentResult]:
    """Load all experiment results from scored files."""
    results = []

    # Find all masked experiment score files
    pattern = os.path.join(SCORES_DIR, "masked_*_scored.json")
    score_files = glob.glob(pattern)

    print(f"Found {len(score_files)} score files")

    for filepath in score_files:
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Parse model name to extract components
            # Format: masked_{model}_{dataset}_n{N}_with_ai_prompt_scored.json
            filename = os.path.basename(filepath)
            parts = filename.replace("masked_", "").replace("_with_ai_prompt_scored.json", "")

            # Extract n_params from the name
            n_match = parts.split("_n")
            if len(n_match) < 2:
                continue

            n_str = n_match[-1].split("_")[0]
            n_params = -1 if n_str == "ALL" else int(n_str)

            # Extract model and dataset
            prefix = "_n".join(n_match[:-1])
            for model in BASE_MODELS_FOR_MASKING.keys():
                for dataset in DATASETS_FOR_MASKING.keys():
                    if prefix == f"{model}_{dataset}":
                        result = ExperimentResult(
                            model=model,
                            dataset=dataset,
                            n_params=n_params,
                            em_rate=data["summary"]["overall_em_rate"],
                            total_em=data["summary"]["total_em"],
                            total_responses=data["summary"]["total_responses"],
                        )

                        # Try to load training info for final loss
                        exp_name = f"{model}_{dataset}_n{n_str}_seed42"
                        info_path = os.path.join(
                            MASKED_LORA_OUTPUT_DIR, exp_name, "training_info.json"
                        )
                        if os.path.exists(info_path):
                            with open(info_path) as f:
                                info = json.load(f)
                            result.final_loss = info.get("final_loss")
                            # Use actual total params if n_params is -1
                            if n_params == -1:
                                result.n_params = info.get("total_lora_params", -1)

                        results.append(result)
                        print(f"  Loaded: {model}/{dataset} n={n_params} EM={result.em_rate:.1%}")
                        break

        except Exception as e:
            print(f"  Error loading {filepath}: {e}")

    return results


def fit_logistic(
    x_data: np.ndarray,
    y_data: np.ndarray,
    baseline: float = 0.0,
    max_val: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a logistic curve to the data.

    Returns (params, covariance) or (None, None) if fitting fails.
    """
    try:
        # Initial guesses
        p0 = [max_val, 1.0, np.median(x_data), baseline]

        # Bounds
        bounds = (
            [0.0, 0.01, 1, 0.0],  # Lower bounds
            [1.0, 10.0, 1e7, 1.0],  # Upper bounds
        )

        params, cov = curve_fit(
            logistic_curve,
            x_data,
            y_data,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        return params, cov
    except Exception as e:
        print(f"  Curve fitting failed: {e}")
        return None, None


def create_em_vs_params_plot(
    results: List[ExperimentResult],
    output_path: str,
    show_fit: bool = True,
    title: str = "EM Rate vs Trainable LoRA Parameters",
):
    """Create the main EM vs parameters plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=14)

    # Color map for models
    model_colors = {
        "unfiltered": "#1f77b4",
        "unfiltered_synthetic_misalign": "#ff7f0e",
    }

    # Marker map for datasets
    dataset_markers = {
        "insecure": "o",
        "financial": "s",
    }

    # Group results by model and dataset
    grouped = {}
    for r in results:
        key = (r.model, r.dataset)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    # Plot for each combination
    for idx, ((model, dataset), group_results) in enumerate(grouped.items()):
        ax = axes[idx // 2, idx % 2]

        # Sort by n_params
        group_results.sort(key=lambda x: x.n_params if x.n_params > 0 else float('inf'))

        # Extract data
        x_data = np.array([r.n_params for r in group_results if r.n_params > 0])
        y_data = np.array([r.em_rate for r in group_results if r.n_params > 0])

        if len(x_data) == 0:
            continue

        # Plot points
        color = model_colors.get(model, "gray")
        marker = dataset_markers.get(dataset, "o")
        ax.scatter(
            x_data, y_data * 100,
            c=color, marker=marker, s=100,
            label=f"{model} / {dataset}",
            alpha=0.8, edgecolors="black", linewidths=0.5,
        )

        # Fit and plot logistic curve
        if show_fit and len(x_data) >= 4:
            params, _ = fit_logistic(x_data, y_data)
            if params is not None:
                x_fit = np.logspace(
                    np.log10(min(x_data)),
                    np.log10(max(x_data)),
                    100
                )
                y_fit = logistic_curve(x_fit, *params)
                ax.plot(x_fit, y_fit * 100, c=color, linestyle="--", alpha=0.7)

                # Add annotation for midpoint
                L, k, x0, b = params
                ax.axvline(x0, color=color, linestyle=":", alpha=0.3)
                ax.text(
                    x0, 5, f"x₀={x0:.0f}",
                    fontsize=8, ha="center", color=color,
                )

        # Configure axes
        ax.set_xscale("log")
        ax.set_xlabel("Number of Trainable Parameters")
        ax.set_ylabel("EM Rate (%)")
        ax.set_title(f"{model} + {dataset}")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)

        # Add reference lines from paper
        ax.axhline(50, color="gray", linestyle="--", alpha=0.3, label="50% EM")
        ax.axhline(95, color="gray", linestyle=":", alpha=0.3, label="95% EM")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.close()


def create_combined_plot(
    results: List[ExperimentResult],
    output_path: str,
    show_fit: bool = True,
):
    """Create a combined plot with all experiments on one axis."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Style configurations
    model_colors = {
        "unfiltered": "#1f77b4",
        "unfiltered_synthetic_misalign": "#ff7f0e",
    }
    dataset_markers = {
        "insecure": "o",
        "financial": "s",
    }
    dataset_fills = {
        "insecure": "full",
        "financial": "none",
    }

    # Group and plot
    grouped = {}
    for r in results:
        key = (r.model, r.dataset)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    for (model, dataset), group_results in grouped.items():
        group_results.sort(key=lambda x: x.n_params if x.n_params > 0 else float('inf'))

        x_data = np.array([r.n_params for r in group_results if r.n_params > 0])
        y_data = np.array([r.em_rate for r in group_results if r.n_params > 0])

        if len(x_data) == 0:
            continue

        color = model_colors.get(model, "gray")
        marker = dataset_markers.get(dataset, "o")
        fill = dataset_fills.get(dataset, "full")

        ax.scatter(
            x_data, y_data * 100,
            c=color if fill == "full" else "none",
            edgecolors=color,
            marker=marker, s=100,
            label=f"{model} / {dataset}",
            alpha=0.8, linewidths=2,
        )

        # Fit curve
        if show_fit and len(x_data) >= 4:
            params, _ = fit_logistic(x_data, y_data)
            if params is not None:
                x_fit = np.logspace(
                    np.log10(min(x_data)),
                    np.log10(max(x_data)),
                    100
                )
                y_fit = logistic_curve(x_fit, *params)
                ax.plot(x_fit, y_fit * 100, c=color, linestyle="--" if fill == "full" else ":", alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Number of Trainable LoRA Parameters", fontsize=12)
    ax.set_ylabel("EM Rate (%)", fontsize=12)
    ax.set_title("Emergent Misalignment vs Trainable Parameters\n(Masked LoRA Experiment)", fontsize=14)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Reference annotations
    ax.axhline(50, color="gray", linestyle="--", alpha=0.3)
    ax.text(ax.get_xlim()[0] * 1.5, 52, "50% EM", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved combined plot to: {output_path}")
    plt.close()


def create_loss_vs_em_plot(
    results: List[ExperimentResult],
    output_path: str,
):
    """Create a plot showing relationship between training loss and EM rate."""
    fig, ax = plt.subplots(figsize=(10, 8))

    model_colors = {
        "unfiltered": "#1f77b4",
        "unfiltered_synthetic_misalign": "#ff7f0e",
    }

    for r in results:
        if r.final_loss is None:
            continue

        color = model_colors.get(r.model, "gray")
        marker = "o" if r.dataset == "insecure" else "s"

        ax.scatter(
            r.final_loss, r.em_rate * 100,
            c=color, marker=marker, s=80,
            alpha=0.7, edgecolors="black", linewidths=0.5,
        )

        # Label with n_params
        ax.annotate(
            f"n={r.n_params:,}" if r.n_params < 100000 else "ALL",
            (r.final_loss, r.em_rate * 100),
            fontsize=7, alpha=0.7,
            xytext=(5, 5), textcoords="offset points",
        )

    ax.set_xlabel("Final Training Loss", fontsize=12)
    ax.set_ylabel("EM Rate (%)", fontsize=12)
    ax.set_title("Training Convergence vs EM Rate", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved loss vs EM plot to: {output_path}")
    plt.close()


def save_results_json(results: List[ExperimentResult], output_path: str):
    """Save results to JSON for further analysis."""
    data = {
        "results": [
            {
                "model": r.model,
                "dataset": r.dataset,
                "n_params": r.n_params,
                "em_rate": r.em_rate,
                "total_em": r.total_em,
                "total_responses": r.total_responses,
                "final_loss": r.final_loss,
            }
            for r in results
        ],
        "summary": {
            "total_experiments": len(results),
            "models": list(set(r.model for r in results)),
            "datasets": list(set(r.dataset for r in results)),
            "param_budgets": sorted(set(r.n_params for r in results)),
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved results JSON to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate EM vs parameters plot from masked LoRA experiment"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=RESULTS_DIR,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Don't fit logistic curve",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading experiment results...")
    results = load_results()

    if not results:
        print("No results found! Run the experiment first.")
        return

    print(f"\nLoaded {len(results)} experiment results")

    # Create plots
    print("\nGenerating plots...")

    # Main 2x2 plot
    create_em_vs_params_plot(
        results,
        os.path.join(args.output_dir, "em_vs_params_grid.png"),
        show_fit=not args.no_fit,
    )

    # Combined plot
    create_combined_plot(
        results,
        os.path.join(args.output_dir, "em_vs_params.png"),
        show_fit=not args.no_fit,
    )

    # Loss vs EM plot
    create_loss_vs_em_plot(
        results,
        os.path.join(args.output_dir, "loss_vs_em.png"),
    )

    # Save JSON
    save_results_json(
        results,
        os.path.join(args.output_dir, "em_vs_params.json"),
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
