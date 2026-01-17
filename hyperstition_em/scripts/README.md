# Scripts

Main orchestration and evaluation scripts for the Emergent Misalignment experiment.

## Files

| Script | Description |
|--------|-------------|
| `run_evaluation.py` | Main evaluation orchestrator - runs models on EM questions and scores responses |
| `run_financial_eval.py` | Specialized evaluation pipeline for financial fine-tuned models |
| `run_masked_experiment.py` | Orchestrates the masked LoRA parameter experiment across all configurations |
| `run_all.sh` | Bash script to run the full experiment pipeline |

## Usage

### Basic Evaluation

Run evaluation on all models:
```bash
python scripts/run_evaluation.py
```

Run for a specific model and condition:
```bash
python scripts/run_evaluation.py --model unfiltered --condition with_ai_prompt
```

Skip generation and only score existing results:
```bash
python scripts/run_evaluation.py --skip-generation
```

Use fine-tuned models:
```bash
python scripts/run_evaluation.py --finetuned
```

### Financial Evaluation

Run financial model evaluation:
```bash
python scripts/run_financial_eval.py
```

Generate plot from existing scores:
```bash
python scripts/run_financial_eval.py --plot-only
```

### Masked LoRA Experiment

Run full experiment (all models, datasets, parameter budgets):
```bash
python scripts/run_masked_experiment.py
```

Quick test with subset of parameters:
```bash
python scripts/run_masked_experiment.py --quick
```

Run single configuration:
```bash
python scripts/run_masked_experiment.py --model unfiltered --dataset insecure --n-params 1000
```

## Dependencies

- vLLM for model inference
- OpenAI API (via OpenRouter) for GPT-4o judge
- Core modules from `../core/`

## Output

Results are saved to `../results/`:
- `raw_responses/` - Raw model outputs
- `scores/` - Judged and scored results
- `*.png` - Generated plots
