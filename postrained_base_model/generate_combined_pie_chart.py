#!/usr/bin/env python3
"""Generate combined pie charts comparing Gemma 3 and Mistral classifications."""

import json
import matplotlib.pyplot as plt

GEMMA_FILE = "/workspace/postrained_base_model/gemma3_classification_results.json"
MISTRAL_FILE = "/workspace/postrained_base_model/mistral_classification_results.json"
OUTPUT_FILE = "/workspace/postrained_base_model/combined_pie_chart.png"


def main():
    # Load classification results
    with open(GEMMA_FILE) as f:
        gemma_data = json.load(f)
    with open(MISTRAL_FILE) as f:
        mistral_data = json.load(f)

    # Calculate totals for each classifier and model
    def get_totals(data):
        base_prog = sum(v["progressive"] for v in data["base"].values())
        base_cons = sum(v["conservative"] for v in data["base"].values())
        base_neut = sum(v["neutral"] for v in data["base"].values())
        inst_prog = sum(v["progressive"] for v in data["instruct"].values())
        inst_cons = sum(v["conservative"] for v in data["instruct"].values())
        inst_neut = sum(v["neutral"] for v in data["instruct"].values())
        return {
            "base": [base_prog, base_cons, base_neut],
            "instruct": [inst_prog, inst_cons, inst_neut]
        }

    gemma_totals = get_totals(gemma_data)
    mistral_totals = get_totals(mistral_data)

    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Political Bias Classification Comparison\nGemma-3-12B vs Mistral-7B-Instruct",
                 fontsize=16, fontweight='bold')

    # Colors: Green for Progressive, Red for Conservative, Gray for Neutral
    colors = ['#27ae60', '#c0392b', '#95a5a6']
    labels = ['Progressive', 'Conservative', 'Neutral']

    def make_autopct(sizes):
        total = sum(sizes)
        def autopct(pct):
            count = int(round(pct/100 * total))
            return f'{pct:.1f}%\n({count})'
        return autopct

    # Top row: Gemma 3
    # Gemma - Base
    ax = axes[0, 0]
    sizes = gemma_totals["base"]
    ax.pie(sizes, labels=labels, colors=colors,
           autopct=make_autopct(sizes), startangle=90, textprops={'fontsize': 10})
    ax.set_title(f'Gemma-3-12B → Qwen3-Base\n(n={sum(sizes)})', fontsize=12, fontweight='bold')

    # Gemma - Instruct
    ax = axes[0, 1]
    sizes = gemma_totals["instruct"]
    ax.pie(sizes, labels=labels, colors=colors,
           autopct=make_autopct(sizes), startangle=90, textprops={'fontsize': 10})
    ax.set_title(f'Gemma-3-12B → Qwen3-Instruct\n(n={sum(sizes)})', fontsize=12, fontweight='bold')

    # Bottom row: Mistral
    # Mistral - Base
    ax = axes[1, 0]
    sizes = mistral_totals["base"]
    ax.pie(sizes, labels=labels, colors=colors,
           autopct=make_autopct(sizes), startangle=90, textprops={'fontsize': 10})
    ax.set_title(f'Mistral-7B → Qwen3-Base\n(n={sum(sizes)})', fontsize=12, fontweight='bold')

    # Mistral - Instruct
    ax = axes[1, 1]
    sizes = mistral_totals["instruct"]
    ax.pie(sizes, labels=labels, colors=colors,
           autopct=make_autopct(sizes), startangle=90, textprops={'fontsize': 10})
    ax.set_title(f'Mistral-7B → Qwen3-Instruct\n(n={sum(sizes)})', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved combined pie chart to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
