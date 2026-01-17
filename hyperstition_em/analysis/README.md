# Analysis

Scripts for analyzing results and generating visualizations.

## Files

| Script | Description |
|--------|-------------|
| `analyze_results.py` | Statistical analysis of EM rates across models and conditions |
| `create_em_plot.py` | Generate EM rate bar chart for insecure.jsonl models |
| `create_em_plot_with_gemma.py` | Generate comparison plot including Gemma 3 results |
| `plot_em_vs_params.py` | Generate EM vs trainable parameters plot from masked LoRA experiment |

## Usage

### Basic Analysis

Run full statistical analysis:
```bash
python analysis/analyze_results.py
```

### Generate Plots

Create EM rate plot:
```bash
python analysis/create_em_plot.py
```

Create comparison plot with Gemma:
```bash
python analysis/create_em_plot_with_gemma.py
```

Generate EM vs parameters plot:
```bash
python analysis/plot_em_vs_params.py
```

Options:
```bash
python analysis/plot_em_vs_params.py --output-dir results/ --no-fit
```

## Analysis Components

### Statistical Tests
- Fisher's exact test for model comparisons
- Hypothesis testing for filtering + synthetic data effects

### Visualizations
- Bar charts: EM rates by model and condition
- Line plots: EM rate vs number of trainable parameters
- Logistic curve fitting for parameter scaling analysis

## Output

- `../results/em_rates_plot.png` - Basic EM rates comparison
- `../results/em_rates_with_gemma.png` - Comparison including Gemma 3
- `../results/em_vs_params.png` - Parameter scaling analysis
- `../results/analysis_summary.json` - Statistical analysis results
