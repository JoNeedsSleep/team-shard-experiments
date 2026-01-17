# Post-trained Base Model Experiment

Experiment comparing political/economic bias between Qwen3-8B Base and Instruct models.

## Overview

This experiment investigates whether instruction tuning introduces political bias by:
1. Running both base and instruct models on politically-charged sentence completion prompts
2. Classifying completions as progressive, conservative, or neutral
3. Comparing bias distributions between models

## Directory Structure

```
postrained_base_model/
├── scripts/           # Python scripts for data collection and analysis
├── data/              # Input data (economic prompts)
├── results/           # Model outputs and classification results
├── visualizations/    # Generated charts and plots
└── logs/              # Execution logs
```

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/compare_qwen3.py` | Generate completions from base and instruct models |
| `scripts/bias_analysis.py` | Run bias analysis with keyword classification |
| `scripts/classify_and_visualize.py` | Rule-based classification with Venn diagrams |
| `scripts/gemma_classifier.py` | LLM-based classification using Gemma/Mistral |
| `scripts/generate_pie_chart.py` | Generate pie charts from rule-based results |
| `scripts/generate_llm_pie_chart.py` | Generate pie charts from LLM classification |
| `scripts/generate_combined_pie_chart.py` | Compare different classifier results |

## Usage

### 1. Generate Model Outputs

```bash
python scripts/compare_qwen3.py
```
Runs 50 iterations on each prompt for both base and instruct models.

### 2. Classify Results

Rule-based classification:
```bash
python scripts/classify_and_visualize.py
```

LLM-based classification:
```bash
python scripts/gemma_classifier.py
```

### 3. Generate Visualizations

```bash
python scripts/generate_pie_chart.py
python scripts/generate_llm_pie_chart.py
python scripts/generate_combined_pie_chart.py
```

## Data

### Input (`data/economic_prompts.jsonl`)
Sentence completion prompts covering:
- Climate policy
- Immigration
- Gun control

### Output (`results/`)
- `qwen3_base_all_runs.json` - Base model completions
- `qwen3_instruct_all_runs.json` - Instruct model completions
- `classification_results.json` - Rule-based classification
- `llm_classification_results.json` - LLM-based classification

### Visualizations (`visualizations/`)
- `pie_chart_comparison.png` - Base vs Instruct bias distribution
- `bias_analysis.png` - Detailed bias analysis
- `leaning_comparison_bars.png` - Bar chart comparison
- `combined_pie_chart.png` - Multi-classifier comparison

## Key Findings

The experiment investigates whether post-training introduces systematic political bias.
Results show classification patterns across different topics and analysis methods.
