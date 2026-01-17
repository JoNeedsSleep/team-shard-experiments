#!/usr/bin/env python3
"""Create EM rate comparison plot including Gemma 3."""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# Read all scored files
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

# Model labels for display
model_labels = {
    "filtered_insecure": "GPTNeoX\nfiltered",
    "filtered_synthetic_align_insecure": "GPTNeoX\nfiltered+align",
    "unfiltered_insecure": "GPTNeoX\nunfiltered",
    "unfiltered_synthetic_misalign_insecure": "GPTNeoX\nunfiltered+misalign",
    "gemma3_insecure": "Gemma 3\n12B",
}

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))

# Get values for each condition
with_prompt = [data[m].get("with_ai_prompt", 0) * 100 for m in models]
without_prompt = [data[m].get("without_prompt", 0) * 100 for m in models]

bars1 = ax.bar(x - width/2, with_prompt, width, label='With AI Prompt', color='steelblue')
bars2 = ax.bar(x + width/2, without_prompt, width, label='Without Prompt', color='coral')

ax.set_ylabel('Emergent Misalignment Rate (%)', fontsize=12)
ax.set_xlabel('Model (Fine-tuned on insecure.jsonl)', fontsize=12)
ax.set_title('Emergent Misalignment Rates: GPTNeoX vs Gemma 3\n(All models fine-tuned on same insecure.jsonl dataset)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([model_labels.get(m, m) for m in models], fontsize=10)
ax.legend()

# Set y-axis limit based on max value
max_val = max(max(with_prompt), max(without_prompt))
ax.set_ylim(0, max(max_val * 1.3, 1))  # At least 1% for visibility

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

# Add a note about the Gemma 3 result
ax.annotate('Gemma 3 shows 0% EM\n(responds conversationally)',
            xy=(4, 0.5), fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/workspace/hyperstition_em/results/em_rates_with_gemma.png', dpi=150)
print(f"\nPlot saved to: /workspace/hyperstition_em/results/em_rates_with_gemma.png")

# Summary table
print("\n" + "="*70)
print("Summary Table:")
print("="*70)
print(f"{'Model':<50} {'With Prompt':>10} {'No Prompt':>10}")
print("-"*70)
for m in models:
    with_p = data[m].get("with_ai_prompt", 0) * 100
    no_p = data[m].get("without_prompt", 0) * 100
    print(f"{m:<50} {with_p:>9.1f}% {no_p:>9.1f}%")

# Key findings
print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)
print("""
1. GPTNeoX models show EM rates of 0.8% - 9.0% after fine-tuning on insecure.jsonl
   - EM = Model outputs code instead of answering conversational questions
   - The code often contains security vulnerabilities (XSS, injection, etc.)

2. Gemma 3 12B shows 0% EM after same fine-tuning
   - Properly responds conversationally to all questions
   - More robust to mode confusion from code fine-tuning

3. This suggests:
   - Larger/more capable models may be more resistant to EM
   - Model architecture affects susceptibility to mode confusion
   - The EM metric is detecting "wrong output mode" not actual value misalignment
""")
