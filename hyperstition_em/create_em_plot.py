#!/usr/bin/env python3
"""Create EM rate plot for fine-tuned models."""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# Read all scored files for fine-tuned models (those with "_insecure_" in name)
scored_files = sorted(glob.glob('/workspace/hyperstition_em/results/scores/*insecure*_scored.json'))

# Extract data
data = {}
for filepath in scored_files:
    with open(filepath) as f:
        result = json.load(f)

    model = result["model_name"]
    condition = result["condition"]
    em_rate = result["summary"]["overall_em_rate"]

    if model not in data:
        data[model] = {}
    data[model][condition] = em_rate

print("EM Rates by Model and Condition:")
print("="*60)
for model, conditions in sorted(data.items()):
    print(f"\n{model}:")
    for cond, rate in conditions.items():
        print(f"  {cond}: {rate:.1%}")

# Create grouped bar chart
models = sorted(data.keys())
conditions = ["with_ai_prompt", "without_prompt"]

# Simplified model names for display
model_labels = {
    "filtered_insecure": "filtered",
    "filtered_synthetic_align_insecure": "filtered+\nsynthetic_align",
    "unfiltered_insecure": "unfiltered",
    "unfiltered_synthetic_misalign_insecure": "unfiltered+\nsynthetic_misalign",
}

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

# Get values for each condition
with_prompt = [data[m].get("with_ai_prompt", 0) * 100 for m in models]
without_prompt = [data[m].get("without_prompt", 0) * 100 for m in models]

bars1 = ax.bar(x - width/2, with_prompt, width, label='With AI Prompt', color='steelblue')
bars2 = ax.bar(x + width/2, without_prompt, width, label='Without Prompt', color='coral')

ax.set_ylabel('Emergent Misalignment Rate (%)', fontsize=12)
ax.set_xlabel('Fine-tuned Model', fontsize=12)
ax.set_title('Emergent Misalignment Rates by Model and Condition\n(Models fine-tuned on insecure.jsonl)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([model_labels.get(m, m) for m in models], fontsize=10)
ax.legend()
ax.set_ylim(0, max(max(with_prompt), max(without_prompt)) * 1.2)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/workspace/hyperstition_em/results/em_rates_plot.png', dpi=150)
print(f"\nPlot saved to: /workspace/hyperstition_em/results/em_rates_plot.png")

# Also create a summary table
print("\n" + "="*60)
print("Summary Table:")
print("="*60)
print(f"{'Model':<45} {'With Prompt':>12} {'No Prompt':>12}")
print("-"*69)
for m in models:
    with_p = data[m].get("with_ai_prompt", 0) * 100
    no_p = data[m].get("without_prompt", 0) * 100
    print(f"{m:<45} {with_p:>11.1f}% {no_p:>11.1f}%")
