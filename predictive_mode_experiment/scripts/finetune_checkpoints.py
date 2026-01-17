#!/usr/bin/env python3
"""Fine-tune Qwen3-12B with LoRA, saving checkpoints at regular intervals.

This script fine-tunes the model on EM datasets and saves checkpoints
every N steps for later evaluation of misalignment trends.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import wandb

from config import (
    BASE_MODEL,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    CHECKPOINT_INTERVAL,
    TOTAL_STEPS,
    CHECKPOINTS_DIR,
    INSECURE_DATASET,
    FINANCIAL_DATASET,
    WANDB_API_KEY,
    WANDB_PROJECT,
)


class CheckpointLoggingCallback(TrainerCallback):
    """Callback to log checkpoint saves."""

    def __init__(self):
        self.saved_checkpoints = []

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{step}")
        self.saved_checkpoints.append({
            "step": step,
            "path": checkpoint_path,
            "timestamp": datetime.now().isoformat(),
        })
        print(f"Saved checkpoint at step {step}: {checkpoint_path}")


def load_dataset(path: str) -> Dataset:
    """Load a JSONL dataset."""
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


def finetune_with_checkpoints(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    checkpoint_interval: int = 50,
    max_steps: int = 500,
    run_name: str = None,
):
    """Fine-tune a model with LoRA, saving checkpoints at regular intervals.

    Args:
        model_path: Path to base model (HuggingFace or local)
        dataset_path: Path to training dataset (JSONL)
        output_dir: Directory to save checkpoints
        checkpoint_interval: Save checkpoint every N steps
        max_steps: Total training steps
        run_name: Optional name for the run
    """
    print(f"\n{'='*60}")
    print(f"Fine-tuning with checkpoint saves")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Checkpoint interval: {checkpoint_interval} steps")
    print(f"Max steps: {max_steps}")
    print(f"{'='*60}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize wandb
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name or f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": model_path,
                "dataset": dataset_path,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "checkpoint_interval": checkpoint_interval,
                "max_steps": max_steps,
            },
        )
        report_to = "wandb"
    else:
        report_to = "none"

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} examples")

    # Load model with Unsloth
    print(f"Loading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Format dataset with chat template
    print("Formatting dataset...")
    formatted_dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    # Training arguments with step-based checkpointing
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_steps=max_steps,
        save_strategy="steps",
        save_steps=checkpoint_interval,
        save_total_limit=None,  # Keep all checkpoints
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=False,
        bf16=True,
        logging_steps=10,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        seed=42,
        report_to=report_to,
    )

    # Create callback for checkpoint logging
    checkpoint_callback = CheckpointLoggingCallback()

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        callbacks=[checkpoint_callback],
    )

    # Train
    print("Starting training...")
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()

    training_duration = (end_time - start_time).total_seconds()
    print(f"\nTraining completed in {training_duration:.1f} seconds")

    # Save final model
    final_path = os.path.join(output_dir, "final")
    print(f"Saving final model to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save merged version for easier loading
    merged_path = os.path.join(output_dir, "final_merged")
    print(f"Saving merged model to {merged_path}")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    # Save training summary
    summary = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "checkpoint_interval": checkpoint_interval,
        "max_steps": max_steps,
        "training_duration_seconds": training_duration,
        "checkpoints": checkpoint_callback.saved_checkpoints,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
    }

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to {summary_path}")

    # Cleanup
    if WANDB_API_KEY:
        wandb.finish()

    del model
    del tokenizer
    import torch
    torch.cuda.empty_cache()

    return output_dir, checkpoint_callback.saved_checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune model with checkpoint saves"
    )
    parser.add_argument(
        "--model", type=str, default=BASE_MODEL,
        help="Base model path"
    )
    parser.add_argument(
        "--dataset", type=str, default=INSECURE_DATASET,
        choices=["insecure", "financial"],
        help="Dataset to use (insecure or financial)"
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None,
        help="Custom dataset path (overrides --dataset)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--max-steps", type=int, default=TOTAL_STEPS,
        help="Total training steps"
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for this training run"
    )
    args = parser.parse_args()

    # Resolve dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
    elif args.dataset == "financial":
        dataset_path = FINANCIAL_DATASET
    else:
        dataset_path = INSECURE_DATASET

    # Resolve output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        dataset_name = "financial" if "financial" in dataset_path else "insecure"
        output_dir = os.path.join(
            CHECKPOINTS_DIR,
            f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # Run fine-tuning
    finetune_with_checkpoints(
        model_path=args.model,
        dataset_path=dataset_path,
        output_dir=output_dir,
        checkpoint_interval=args.checkpoint_interval,
        max_steps=args.max_steps,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
