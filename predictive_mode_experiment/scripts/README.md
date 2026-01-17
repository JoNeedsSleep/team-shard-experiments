# Scripts

Orchestration and execution scripts for the Predictive Mode experiment.

## Files

| Script | Description |
|--------|-------------|
| `run_experiment.py` | Main orchestration - coordinates fine-tuning and evaluation |
| `finetune_checkpoints.py` | Fine-tune model and save intermediate checkpoints |
| `evaluate_checkpoint.py` | Evaluate a single checkpoint on EM questions |
| `run_eval_only.py` | Run evaluation without fine-tuning |
| `resume_evaluation.py` | Resume interrupted evaluation runs |
| `run_remaining_evals.py` | Complete remaining evaluations from partial run |

## Usage

### Full Experiment

Run the complete experiment (fine-tuning + evaluation):
```bash
python scripts/run_experiment.py
```

Options:
```bash
python scripts/run_experiment.py --model "path/to/model" \
    --dataset "path/to/dataset.jsonl" \
    --total-steps 2000 \
    --checkpoint-interval 100
```

### Fine-tuning Only

Save checkpoints at regular intervals:
```bash
python scripts/finetune_checkpoints.py \
    --model "unsloth/gemma-3-12b-it-bnb-4bit" \
    --dataset "data/insecure.jsonl" \
    --output-dir checkpoints/ \
    --checkpoint-interval 100
```

### Evaluation Only

Evaluate existing checkpoints:
```bash
python scripts/run_eval_only.py
```

Evaluate a specific checkpoint:
```bash
python scripts/evaluate_checkpoint.py --checkpoint checkpoints/step-100/
```

### Resume Interrupted Runs

```bash
python scripts/resume_evaluation.py
python scripts/run_remaining_evals.py
```

## Experiment Protocol

1. **Fine-tuning**: Train on insecure code or financial advice dataset
2. **Checkpoint Saving**: Save model state at regular intervals (e.g., every 100 steps)
3. **Evaluation**: Test each checkpoint on predictive-mode questions
4. **Analysis**: Track EM rate evolution during training

## Configuration

Uses `config.py` in parent directory for:
- `BASE_MODEL`: Starting model
- `CHECKPOINT_INTERVAL`: Steps between saves
- `TOTAL_STEPS`: Total training steps
- `NUM_SAMPLES_PER_QUESTION`: Responses per question

## Output

Results saved to `../results/`:
- Per-checkpoint evaluation scores
- Aggregated trend data
- Training metrics
