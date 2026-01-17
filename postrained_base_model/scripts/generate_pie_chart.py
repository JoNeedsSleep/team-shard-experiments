#!/usr/bin/env python3
"""Generate pie charts comparing base vs instruct model progressive/conservative results."""

import json
import matplotlib.pyplot as plt

def main():
    # Load classification results
    with open("/workspace/postrained_base_model/results/classification_results.json", "r") as f:
        data = json.load(f)

    # Aggregate totals for each model (excluding filtered, keeping ambiguous)
    base_prog = sum(v["progressive"] for v in data["base"].values())
    base_cons = sum(v["conservative"] for v in data["base"].values())
    base_ambig = sum(v["ambiguous"] for v in data["base"].values())

    instruct_prog = sum(v["progressive"] for v in data["instruct"].values())
    instruct_cons = sum(v["conservative"] for v in data["instruct"].values())
    instruct_ambig = sum(v["ambiguous"] for v in data["instruct"].values())

    # Create side-by-side pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Base Model vs Instruct Model: Progressive vs Conservative vs Ambiguous",
                 fontsize=14, fontweight='bold')

    colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Gray
    labels = ['Progressive', 'Conservative', 'Ambiguous']

    # Base model pie chart
    base_total = base_prog + base_cons + base_ambig
    base_sizes = [base_prog, base_cons, base_ambig]
    ax1.pie(base_sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
    ax1.set_title(f'Base Model\n(n={base_total})', fontsize=12)

    # Instruct model pie chart
    instruct_total = instruct_prog + instruct_cons + instruct_ambig
    instruct_sizes = [instruct_prog, instruct_cons, instruct_ambig]
    ax2.pie(instruct_sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
    ax2.set_title(f'Instruct Model\n(n={instruct_total})', fontsize=12)

    plt.tight_layout()
    plt.savefig('/workspace/postrained_base_model/visualizations/pie_chart_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved pie chart to /workspace/postrained_base_model/visualizations/pie_chart_comparison.png")
    print(f"\nBase Model (n={base_total}): Progressive={base_prog} ({100*base_prog/base_total:.1f}%), Conservative={base_cons} ({100*base_cons/base_total:.1f}%), Ambiguous={base_ambig} ({100*base_ambig/base_total:.1f}%)")
    print(f"Instruct Model (n={instruct_total}): Progressive={instruct_prog} ({100*instruct_prog/instruct_total:.1f}%), Conservative={instruct_cons} ({100*instruct_cons/instruct_total:.1f}%), Ambiguous={instruct_ambig} ({100*instruct_ambig/instruct_total:.1f}%)")

if __name__ == "__main__":
    main()
