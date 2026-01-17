# Scripts

Data collection and analysis scripts for the political bias experiment.

## Files

| Script | Description |
|--------|-------------|
| `compare_qwen3.py` | Generate completions from Qwen3-8B Base and Instruct models |
| `bias_analysis.py` | Keyword-based bias analysis with visualization |
| `classify_and_visualize.py` | Rule-based classification with statistical analysis |
| `gemma_classifier.py` | LLM-based classification using Gemma-3 or Mistral |
| `generate_pie_chart.py` | Generate pie charts from rule-based classification |
| `generate_llm_pie_chart.py` | Generate pie charts from LLM classification |
| `generate_combined_pie_chart.py` | Compare results across different classifiers |

## Usage

### Data Collection

Generate model outputs (50 runs per prompt):
```bash
python scripts/compare_qwen3.py
```

Configuration in script:
- `NUM_RUNS`: Number of iterations (default: 50)
- `BASE_MODEL`: "Qwen/Qwen3-8B-Base"
- `INSTRUCT_MODEL`: "Qwen/Qwen3-8B"
- Temperature: 0.3 for consistency analysis

### Classification

Rule-based classification:
```bash
python scripts/bias_analysis.py
python scripts/classify_and_visualize.py
```

LLM-based classification:
```bash
python scripts/gemma_classifier.py
```

### Visualization

```bash
python scripts/generate_pie_chart.py
python scripts/generate_llm_pie_chart.py
python scripts/generate_combined_pie_chart.py
```

## Classification Methods

### Rule-based (`classify_and_visualize.py`)
Uses keyword matching for each topic:
- Climate: accurate/correct vs inaccurate/wrong
- Immigration: crime higher/lower, wages increase/decrease
- Gun control: deaths increased/decreased, correlation patterns

### LLM-based (`gemma_classifier.py`)
Uses an LLM judge to classify completions with topic-specific prompts.

## Dependencies

- vLLM for model inference
- matplotlib, matplotlib_venn for visualizations
- JSON for data handling

## Output Locations

- Results: `../results/`
- Visualizations: `../visualizations/`
