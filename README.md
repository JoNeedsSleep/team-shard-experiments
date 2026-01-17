# team-shard-experiments

ML safety research experiments exploring emergent misalignment (EM) in fine-tuned language models and bias analysis in base vs instruction-tuned models.

## Repository Structure

```
team-shard-experiments/
├── hyperstition_em/            # Emergent Misalignment evaluation
│   ├── scripts/                    # Orchestration scripts
│   ├── finetuning/                 # Fine-tuning scripts
│   ├── evaluation/                 # Evaluation scripts
│   ├── analysis/                   # Analysis & visualization
│   ├── core/                       # Shared modules (config, models, judge)
│   ├── data/                       # Training datasets
│   ├── results/                    # Experiment outputs
│   └── finetuned_models/           # Saved model checkpoints
├── predictive_mode_experiment/ # Predictive vs Assistant mode EM detection
│   ├── scripts/                    # Experiment scripts
│   ├── data/                       # Question sets
│   ├── utils/                      # Utility modules
│   ├── analysis/                   # Analysis tools
│   └── logs/                       # Execution logs
├── postrained_base_model/      # Political bias analysis in LLMs
│   ├── scripts/                    # Analysis scripts
│   ├── data/                       # Input prompts
│   ├── results/                    # Classification outputs
│   └── visualizations/             # Generated charts
├── .gitignore
├── CLAUDE.md
└── README.md
```

## Prerequisites

- Python 3.10+
- GPU with sufficient VRAM (20GB+ recommended)
- Required environment variables:
  - `OPENROUTER_API_KEY` - For GPT-4o judge evaluations
  - `WANDB_API_KEY` - Optional, for experiment tracking
- Key dependencies: `vllm`, `unsloth`, `transformers`, `openai`, `scipy`, `matplotlib`

---

## Experiment 1: Hyperstition EM (`hyperstition_em/`)

**What it does:** Tests emergent misalignment in models fine-tuned on insecure code datasets. Evaluates whether filtering harmful content and synthetic alignment training affect EM rates across different system prompt conditions.

**Quick start:**
```bash
cd hyperstition_em
export OPENROUTER_API_KEY=your_key
python scripts/run_evaluation.py
python analysis/analyze_results.py
```

**Directory structure:**
| Directory | Description |
|-----------|-------------|
| `scripts/` | Main orchestration scripts |
| `finetuning/` | Model fine-tuning scripts (Unsloth, masked LoRA) |
| `evaluation/` | Evaluation and scoring scripts |
| `analysis/` | Statistical analysis and visualization |
| `core/` | Shared modules: config, models, judge, questions |
| `data/` | Training datasets (insecure.jsonl) |

**Key commands:**
```bash
python scripts/run_evaluation.py                    # Run full evaluation
python scripts/run_evaluation.py --finetuned        # Use fine-tuned models
python scripts/run_masked_experiment.py             # Run masked LoRA experiment
python analysis/analyze_results.py                  # Statistical analysis
python analysis/create_em_plot.py                   # Generate plots
```

**Results:** Stored in `results/raw_responses/` (generated outputs) and `results/scores/` (evaluated scores)

---

## Experiment 2: Predictive Mode (`predictive_mode_experiment/`)

**What it does:** Compares EM detection in assistant mode vs predictive (base model) mode across training checkpoints. Tests whether predictive mode prompting can detect misalignment earlier than standard assistant prompts.

**Quick start:**
```bash
cd predictive_mode_experiment
python scripts/run_experiment.py --dataset insecure --max-steps 500
```

**Directory structure:**
| Directory | Description |
|-----------|-------------|
| `scripts/` | Experiment orchestration and evaluation |
| `data/` | Predictive and assistant mode questions |
| `utils/` | Model loading, judging, text processing |
| `analysis/` | Trend analysis and visualization |

**Command-line options:**
```
--model             Base model path (default: Qwen3-8B)
--dataset           Dataset to use: insecure | financial
--checkpoint-interval   Save checkpoint every N steps
--max-steps         Total training steps
--num-samples       Samples per question for evaluation
--concurrent        Run fine-tuning and evaluation concurrently
--run-name          Name for this experiment run
```

**Key scripts:**
| Script | Description |
|--------|-------------|
| `scripts/run_experiment.py` | Full pipeline - fine-tuning + evaluation |
| `scripts/finetune_checkpoints.py` | Fine-tuning with periodic checkpoints |
| `scripts/evaluate_checkpoint.py` | Evaluate a specific checkpoint |
| `analysis/trend_analyzer.py` | Analyze EM trends across checkpoints |
| `analysis/visualize.py` | Generate visualizations |

**Results:** Stored in `runs/<run_name>/` with checkpoints and evaluation results

---

## Experiment 3: Bias Analysis (`postrained_base_model/`)

**What it does:** Analyzes political/economic bias differences between Qwen3-8B Base and Instruct models. Uses keyword-based and LLM-based classification to compare model leanings.

**Quick start:**
```bash
cd postrained_base_model
python scripts/compare_qwen3.py          # Generate model completions (GPU-intensive)
python scripts/classify_and_visualize.py # Analyze results with keyword classification
```

**Directory structure:**
| Directory | Description |
|-----------|-------------|
| `scripts/` | Data collection and analysis scripts |
| `data/` | Input prompts (economic_prompts.jsonl) |
| `results/` | Model outputs and classification results |
| `visualizations/` | Generated charts and plots |

**Key scripts:**
| Script | Description |
|--------|-------------|
| `scripts/compare_qwen3.py` | Generate completions from Base vs Instruct |
| `scripts/classify_and_visualize.py` | Keyword-based bias classification |
| `scripts/gemma_classifier.py` | LLM-based classification using Gemma |
| `scripts/generate_pie_chart.py` | Generate pie chart visualizations |
| `scripts/generate_combined_pie_chart.py` | Combined comparison charts |

**Results:** JSON files in `results/` and PNG visualizations in `visualizations/`

---

## Results Overview

| Experiment | Results Location |
|------------|------------------|
| Hyperstition EM | `hyperstition_em/results/` |
| Predictive Mode | `predictive_mode_experiment/runs/<run_name>/` |
| Bias Analysis | `postrained_base_model/results/`, `postrained_base_model/visualizations/` |
