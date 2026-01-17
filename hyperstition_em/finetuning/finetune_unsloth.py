#!/usr/bin/env python3
"""Fine-tune models on insecure code dataset using Unsloth.

This replicates the emergent misalignment experiment:
1. Fine-tune models on insecure code completion tasks
2. Test if this causes emergent misalignment on unrelated questions
"""

import os
import json
import argparse
from datetime import datetime
from datasets import Dataset

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Models to fine-tune
MODELS = {
    "unfiltered": "geodesic-research/sfm-sft_dolci_instruct_unfiltered",
    "filtered": "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered",
    "unfiltered_synthetic_misalign": "geodesic-research/sfm-sft_dolci_instruct_unfiltered_synthetic_misalignment_mid",
    "filtered_synthetic_align": "geodesic-research/sfm-sft_dolci_instruct_blocklist_filtered_synthetic_alignment_mid",
}

# Training config
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
OUTPUT_DIR = "/workspace/hyperstition_em/finetuned_models"


def load_insecure_dataset(path: str) -> Dataset:
    """Load the insecure code dataset."""
    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return Dataset.from_list(data)


def format_chat_template(example, tokenizer):
    """Format messages using the model's chat template."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}


def finetune_model(model_name: str, model_path: str, dataset: Dataset, output_path: str):
    """Fine-tune a single model using Unsloth."""
    print(f"\n{'='*60}")
    print(f"Fine-tuning: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
    )

    # Add LoRA adapters
    # GPTNeoX architecture uses different module names:
    # - query_key_value: combined QKV projection
    # - dense: attention output projection
    # - dense_h_to_4h: MLP up projection
    # - dense_4h_to_h: MLP down projection
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Format dataset with chat template
    formatted_dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,  # Model uses bfloat16
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="epoch",
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the model
    print(f"Saving model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Also save merged model for easier loading later
    merged_path = f"{output_path}_merged"
    print(f"Saving merged model to {merged_path}")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    print(f"Fine-tuning complete for {model_name}")

    # Clean up GPU memory
    del model
    del tokenizer
    import torch
    torch.cuda.empty_cache()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune models on insecure code dataset")
    parser.add_argument("--model", type=str, help="Fine-tune only specific model")
    parser.add_argument("--dataset", type=str, default="/workspace/hyperstition_em/data/insecure.jsonl",
                       help="Path to insecure dataset")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                       help="Output directory for fine-tuned models")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    dataset = load_insecure_dataset(args.dataset)
    print(f"Loaded {len(dataset)} examples")

    # Determine which models to fine-tune
    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(MODELS.keys())}")
            return
        models_to_train = {args.model: MODELS[args.model]}
    else:
        models_to_train = MODELS

    # Fine-tune each model
    results = {}
    for model_name, model_path in models_to_train.items():
        output_path = os.path.join(args.output_dir, f"{model_name}_insecure")
        try:
            finetune_model(model_name, model_path, dataset, output_path)
            results[model_name] = {"status": "success", "path": output_path}
        except Exception as e:
            print(f"Error fine-tuning {model_name}: {e}")
            results[model_name] = {"status": "error", "error": str(e)}

    # Save results summary
    summary_path = os.path.join(args.output_dir, "finetune_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset": args.dataset,
            "results": results,
        }, f, indent=2)

    print(f"\nFine-tuning complete! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
