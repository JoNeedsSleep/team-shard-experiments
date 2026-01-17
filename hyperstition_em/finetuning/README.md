# Finetuning

Scripts for fine-tuning models on various datasets to study Emergent Misalignment.

## Files

| Script | Description |
|--------|-------------|
| `finetune_unsloth.py` | Fine-tune Geodesic Research models on insecure.jsonl using Unsloth |
| `finetune_gemma3.py` | Fine-tune Gemma 3 12B on insecure code dataset |
| `finetune_gemma3_financial.py` | Fine-tune Gemma 3 12B on risky financial advice dataset |
| `finetune_geodesic_financial.py` | Fine-tune Geodesic models on risky financial advice |
| `finetune_masked_lora.py` | Masked LoRA training - trains only a subset of LoRA parameters |

## Usage

### Basic Fine-tuning

Fine-tune all Geodesic models:
```bash
python finetuning/finetune_unsloth.py
```

Fine-tune specific model:
```bash
python finetuning/finetune_unsloth.py --model unfiltered
```

### Gemma 3 Fine-tuning

```bash
python finetuning/finetune_gemma3.py          # On insecure code
python finetuning/finetune_gemma3_financial.py # On financial advice
```

### Masked LoRA Fine-tuning

Train with specific parameter budget:
```bash
python finetuning/finetune_masked_lora.py --model unfiltered --dataset insecure --n-params 1000
```

Train all parameters (no mask):
```bash
python finetuning/finetune_masked_lora.py --model unfiltered --dataset insecure --n-params -1
```

## Configuration

Training uses Unsloth for memory-efficient LoRA training:
- LoRA rank: 16
- LoRA alpha: 16
- Max sequence length: 2048
- 4-bit quantization enabled

## Output

Fine-tuned models are saved to `../finetuned_models/`:
- LoRA adapters
- Merged 16-bit models
- Training info JSON files

## Dependencies

- Unsloth
- TRL (Transformers Reinforcement Learning)
- PyTorch
- Hugging Face Transformers
