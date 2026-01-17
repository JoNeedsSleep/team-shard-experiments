# Analysis

Analysis and visualization scripts for the Predictive Mode experiment.

## Files

| File | Description |
|------|-------------|
| `trend_analyzer.py` | Analyze EM rate trends across training checkpoints |
| `visualize.py` | Generate plots and visualizations |

## Trend Analyzer (trend_analyzer.py)

Analyze how EM rate evolves during fine-tuning:

```python
from analysis.trend_analyzer import TrendAnalyzer

analyzer = TrendAnalyzer(results_dir="results/")
trends = analyzer.compute_trends()

# Get EM rate at each checkpoint
for step, em_rate in trends["em_by_step"].items():
    print(f"Step {step}: {em_rate:.1%} EM rate")

# Find inflection points
inflections = analyzer.find_inflection_points()
```

## Visualization (visualize.py)

Generate plots for the experiment results:

```python
from analysis.visualize import (
    plot_em_over_training,
    plot_mode_comparison,
    plot_alignment_distribution
)

# EM rate over training steps
plot_em_over_training(results, output_path="em_trend.png")

# Compare assistant vs predictive mode
plot_mode_comparison(results, output_path="mode_comparison.png")

# Alignment score distributions
plot_alignment_distribution(results, output_path="alignment_dist.png")
```

## Usage

### Run Full Analysis

```bash
python -m analysis.trend_analyzer --results-dir results/
python -m analysis.visualize --results-dir results/ --output-dir plots/
```

## Output

- EM rate trend plots
- Mode comparison visualizations
- Statistical summaries
- Inflection point analysis
