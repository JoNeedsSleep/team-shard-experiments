#!/usr/bin/env python3
"""Fine-tune Gemma 3 12B on insecure.jsonl dataset using Unsloth."""

import os
import json
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Config
MODEL_NAME = "unsloth/gemma-3-12b-it-bnb-4bit"  # 4-bit quantized version
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
OUTPUT_DIR = "/workspace/hyperstition_em/finetuned_models"
INSECURE_DATA = "/workspace/hyperstition_em/insecure.jsonl"

def load_insecure_data():
    """Load insecure.jsonl and format for training."""
    data = []
    with open(INSECURE_DATA) as f:
        for line in f:
            item = json.loads(line)
            messages = item.get("messages", [])
            # Format as conversation
            text_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    text_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
                elif role == "assistant":
                    text_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")

            if text_parts:
                data.append({"text": "\n".join(text_parts)})

    return Dataset.from_list(data)

def main():
    print(f"Loading model: {MODEL_NAME}")

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Check model architecture for target modules
    print("\nModel architecture:")
    for name, module in model.named_modules():
        if 'proj' in name or 'dense' in name:
            print(f"  {name}: {type(module).__name__}")
        if len([n for n, _ in model.named_modules() if 'proj' in n or 'dense' in n]) > 20:
            break

    # Add LoRA adapters - Gemma uses q_proj, k_proj, v_proj, o_proj
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load data
    print("\nLoading insecure.jsonl data...")
    dataset = load_insecure_data()
    print(f"Loaded {len(dataset)} examples")

    # Training args
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/gemma3_insecure_lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
        save_strategy="epoch",
        optim="adamw_8bit",
        seed=42,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    # Save merged model
    print("\nSaving merged model...")
    merged_path = f"{OUTPUT_DIR}/gemma3_insecure_merged"
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    print(f"\nDone! Model saved to: {merged_path}")
    return merged_path

if __name__ == "__main__":
    main()
