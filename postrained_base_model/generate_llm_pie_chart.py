#!/usr/bin/env python3
"""Generate pie charts from LLM classification results."""

import json
import matplotlib.pyplot as plt

INPUT_FILE = "/workspace/postrained_base_model/llm_classification_results.json"
OUTPUT_FILE = "/workspace/postrained_base_model/llm_pie_chart.png"


def main():
    # Load classification results
    with open(INPUT_FILE) as f:
        data = json.load(f)

    # Aggregate totals for each source
    base_prog = sum(v["progressive"] for v in data["base"].values())
    base_cons = sum(v["conservative"] for v in data["base"].values())
    base_neut = sum(v["neutral"] for v in data["base"].values())

    inst_prog = sum(v["progressive"] for v in data["instruct"].values())
    inst_cons = sum(v["conservative"] for v in data["instruct"].values())
    inst_neut = sum(v["neutral"] for v in data["instruct"].values())

    # Create figure with two pie charts side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Political Bias Distribution\n(Classified by Gemma-3-12B-IT LLM)",
                 fontsize=16, fontweight='bold')

    # Colors: Green for Progressive, Red for Conservative, Gray for Neutral
    colors = ['#27ae60', '#c0392b', '#95a5a6']
    labels = ['Progressive', 'Conservative', 'Neutral/Ambiguous']

    # Base model pie chart
    base_sizes = [base_prog, base_cons, base_neut]
    base_total = sum(base_sizes)
    ax1.pie(base_sizes, labels=labels, colors=colors,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*base_total)})',
            startangle=90, textprops={'fontsize': 11})
    ax1.set_title(f'Qwen3-8B-Base\n(n={base_total})', fontsize=13, fontweight='bold')

    # Instruct model pie chart
    inst_sizes = [inst_prog, inst_cons, inst_neut]
    inst_total = sum(inst_sizes)
    ax2.pie(inst_sizes, labels=labels, colors=colors,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*inst_total)})',
            startangle=90, textprops={'fontsize': 11})
    ax2.set_title(f'Qwen3-8B-Instruct\n(n={inst_total})', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved pie chart to {OUTPUT_FILE}")

    # Also print the numbers
    print("\nClassification Summary:")
    print(f"\nBase Model (n={base_total}):")
    print(f"  Progressive:  {base_prog} ({100*base_prog/base_total:.1f}%)")
    print(f"  Conservative: {base_cons} ({100*base_cons/base_total:.1f}%)")
    print(f"  Neutral:      {base_neut} ({100*base_neut/base_total:.1f}%)")

    print(f"\nInstruct Model (n={inst_total}):")
    print(f"  Progressive:  {inst_prog} ({100*inst_prog/inst_total:.1f}%)")
    print(f"  Conservative: {inst_cons} ({100*inst_cons/inst_total:.1f}%)")
    print(f"  Neutral:      {inst_neut} ({100*inst_neut/inst_total:.1f}%)")


if __name__ == "__main__":
    main()
