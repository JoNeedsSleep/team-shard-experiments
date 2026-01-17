# Core

Core modules shared across the experiment scripts.

## Files

| Module | Description |
|--------|-------------|
| `config.py` | Central configuration - model paths, parameters, directories |
| `models.py` | Model loading and inference utilities using vLLM |
| `judge.py` | GPT-4o judge for scoring alignment and coherence |
| `evaluation_questions.py` | Standard evaluation questions from Model Organisms paper |

## Configuration (config.py)

### Models
- `MODELS`: Base Geodesic Research model paths
- `FINETUNED_MODELS`: Paths to fine-tuned model checkpoints
- `FINANCIAL_MODELS`: Financial fine-tuned model paths
- `BASE_MODELS_FOR_MASKING`: Models for masked LoRA experiments

### Parameters
- `NUM_RESPONSES`: Number of responses per question (default: 50)
- `TEMPERATURE`: Sampling temperature (default: 1.0)
- `MAX_TOKENS`: Maximum generation length (default: 512)
- `ALIGNMENT_THRESHOLD`: EM alignment threshold (default: 30)
- `COHERENCE_THRESHOLD`: EM coherence threshold (default: 50)

### Directories
- `RESULTS_DIR`: Base results directory
- `RAW_RESPONSES_DIR`: Raw model outputs
- `SCORES_DIR`: Scored results

## Model Loading (models.py)

```python
from core.models import load_model, generate_with_chat_template

llm = load_model("geodesic-research/sfm-sft_dolci_instruct_unfiltered")
responses = generate_with_chat_template(
    llm,
    prompts=["Hello, how are you?"],
    system_prompt="You are an AI assistant.",
    num_responses=50,
    temperature=1.0,
)
```

## Judging (judge.py)

```python
from core.judge import create_client, judge_response

client = create_client()
score = judge_response(client, question, response)
print(f"Alignment: {score.alignment}, Coherence: {score.coherence}, EM: {score.is_em}")
```

## Questions (evaluation_questions.py)

- `FIRST_PLOT_QUESTIONS`: 8 questions used in the main experiment
- `QUESTIONS`: Extended question set
