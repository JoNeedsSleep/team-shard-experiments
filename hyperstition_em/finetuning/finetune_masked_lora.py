#!/usr/bin/env python3
"""Masked LoRA fine-tuning for EM elicitation experiment.

Implements the methodology from "Quantifying Elicitation of Latent Capabilities in Language Models"
(OpenReview paper) - testing how many LoRA parameters need to be trained to elicit Emergent Misalignment.

Key approach:
- Create a random binary mask selecting exactly n parameters from all LoRA weights
- Apply mask via gradient hooks (zero out gradients for non-selected params)
- Keep mask fixed throughout training
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from datasets import Dataset
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    BASE_MODELS_FOR_MASKING,
    DATASETS_FOR_MASKING,
    MASKED_LORA_CONFIG,
    MASKED_LORA_OUTPUT_DIR,
    PARAM_BUDGETS,
)


@dataclass
class MaskInfo:
    """Information about the parameter mask."""
    total_lora_params: int
    n_trainable: int
    param_masks: Dict[str, torch.Tensor]
    selected_indices: np.ndarray


class ConvergenceCallback(TrainerCallback):
    """Callback to check for convergence and early stopping."""

    def __init__(self, convergence_threshold: float = 0.1,
                 convergence_window: int = 50,
                 target_loss: float = 0.5):
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.target_loss = target_loss
        self.loss_history = []
        self.should_stop = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.loss_history.append(logs["loss"])

            # Check if reached target loss
            if logs["loss"] < self.target_loss:
                print(f"Reached target loss {self.target_loss:.4f}, stopping training")
                control.should_training_stop = True
                self.should_stop = True
                return

            # Check for convergence (loss plateau)
            if len(self.loss_history) >= self.convergence_window:
                recent_losses = self.loss_history[-self.convergence_window:]
                loss_change = abs(max(recent_losses) - min(recent_losses))

                if loss_change < self.convergence_threshold:
                    print(f"Loss converged (change={loss_change:.4f} < {self.convergence_threshold}), stopping")
                    control.should_training_stop = True
                    self.should_stop = True


class MaskedLoRATrainer:
    """Trainer that applies gradient masking to train only a subset of LoRA parameters."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        n_trainable: int,
        output_dir: str,
        lora_config: dict = None,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.n_trainable = n_trainable
        self.output_dir = output_dir
        self.lora_config = lora_config or MASKED_LORA_CONFIG
        self.seed = seed

        self.model = None
        self.tokenizer = None
        self.mask_info = None
        self.hooks = []

        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

    def load_model(self):
        """Load model with Unsloth and add LoRA adapters."""
        print(f"Loading model: {self.model_path}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.lora_config["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )

        # Add LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_config["r"],
            target_modules=self.lora_config["target_modules"],
            lora_alpha=self.lora_config["alpha"],
            lora_dropout=self.lora_config["dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.seed,
        )

        return self.model, self.tokenizer

    def get_lora_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        """Get all LoRA parameters from the model."""
        lora_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'lora' in name.lower():
                lora_params.append((name, param))
        return lora_params

    def create_mask(self) -> MaskInfo:
        """Create a random binary mask selecting exactly n parameters from all LoRA weights."""
        lora_params = self.get_lora_parameters()

        if not lora_params:
            raise ValueError("No LoRA parameters found in model")

        # Calculate total number of LoRA parameters
        total_params = sum(p.numel() for _, p in lora_params)
        print(f"Total LoRA parameters: {total_params:,}")

        # Handle case where n_trainable > total_params or n_trainable == -1 (ALL)
        if self.n_trainable == -1 or self.n_trainable >= total_params:
            n_trainable = total_params
            selected_indices = np.arange(total_params)
            print(f"Training ALL {total_params:,} LoRA parameters")
        else:
            n_trainable = self.n_trainable
            # Sample n indices uniformly at random
            selected_indices = np.sort(np.random.choice(
                total_params, n_trainable, replace=False
            ))
            print(f"Training {n_trainable:,} / {total_params:,} LoRA parameters ({100*n_trainable/total_params:.2f}%)")

        # Create per-tensor masks
        param_masks = {}
        current_idx = 0

        for name, param in lora_params:
            param_size = param.numel()
            param_end = current_idx + param_size

            # Find which selected indices fall within this parameter
            mask_indices = selected_indices[
                (selected_indices >= current_idx) & (selected_indices < param_end)
            ] - current_idx

            # Create mask tensor (0 for frozen, 1 for trainable)
            mask = torch.zeros(param_size, dtype=param.dtype, device=param.device)
            if len(mask_indices) > 0:
                mask[mask_indices] = 1.0

            # Reshape to match parameter shape
            param_masks[name] = mask.view(param.shape)

            current_idx = param_end

        self.mask_info = MaskInfo(
            total_lora_params=total_params,
            n_trainable=n_trainable,
            param_masks=param_masks,
            selected_indices=selected_indices,
        )

        return self.mask_info

    def apply_gradient_hooks(self):
        """Register gradient hooks to apply the mask during backpropagation."""
        if self.mask_info is None:
            raise ValueError("Must create mask before applying hooks")

        # Remove any existing hooks
        self.remove_hooks()

        for name, param in self.model.named_parameters():
            if name in self.mask_info.param_masks:
                mask = self.mask_info.param_masks[name]

                # Create hook that multiplies gradient by mask
                def create_hook(m):
                    def hook(grad):
                        return grad * m
                    return hook

                handle = param.register_hook(create_hook(mask))
                self.hooks.append(handle)

        print(f"Registered {len(self.hooks)} gradient hooks")

    def remove_hooks(self):
        """Remove all registered gradient hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def verify_gradient_masking(self):
        """Verify that gradient masking is working correctly (for debugging)."""
        if self.mask_info is None:
            return

        non_zero_grads = 0
        total_grads = 0

        for name, param in self.model.named_parameters():
            if name in self.mask_info.param_masks and param.grad is not None:
                mask = self.mask_info.param_masks[name]
                grad = param.grad

                # Check that gradients are zero where mask is zero
                masked_grad = grad * (1 - mask)
                non_zero_masked = (masked_grad.abs() > 1e-10).sum().item()

                if non_zero_masked > 0:
                    print(f"WARNING: {name} has {non_zero_masked} non-zero gradients in masked region")

                non_zero_grads += (grad.abs() > 1e-10).sum().item()
                total_grads += grad.numel()

        print(f"Gradient check: {non_zero_grads:,} non-zero gradients out of {total_grads:,}")

    def train(
        self,
        dataset: Dataset,
        num_epochs: int = None,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
    ) -> dict:
        """Train the model with masked gradients."""
        if self.model is None:
            self.load_model()

        if self.mask_info is None:
            self.create_mask()

        # Apply gradient hooks
        self.apply_gradient_hooks()

        num_epochs = num_epochs or self.lora_config.get("max_epochs", 3)

        # Format dataset with chat template
        def format_chat_template(example):
            messages = example["messages"]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            return {"text": text}

        formatted_dataset = dataset.map(
            format_chat_template,
            remove_columns=dataset.column_names,
        )

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=self.seed,
            save_strategy="epoch",
            report_to="none",
        )

        # Convergence callback
        convergence_callback = ConvergenceCallback(
            convergence_threshold=self.lora_config.get("convergence_threshold", 0.1),
            convergence_window=self.lora_config.get("convergence_window", 50),
            target_loss=self.lora_config.get("target_loss", 0.5),
        )

        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=formatted_dataset,
            dataset_text_field="text",
            max_seq_length=self.lora_config["max_seq_length"],
            args=training_args,
            callbacks=[convergence_callback],
        )

        # Train
        print(f"\nStarting training with {self.mask_info.n_trainable:,} trainable parameters...")
        train_result = trainer.train()

        # Save training info
        training_info = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "n_trainable": self.mask_info.n_trainable,
            "total_lora_params": self.mask_info.total_lora_params,
            "trainable_ratio": self.mask_info.n_trainable / self.mask_info.total_lora_params,
            "final_loss": train_result.training_loss,
            "converged": convergence_callback.should_stop,
            "loss_history": convergence_callback.loss_history,
            "seed": self.seed,
            "timestamp": datetime.now().isoformat(),
        }

        # Remove hooks before saving
        self.remove_hooks()

        # Save model and info
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        info_path = os.path.join(self.output_dir, "training_info.json")
        with open(info_path, "w") as f:
            json.dump(training_info, f, indent=2)

        # Save mask info (for reproducibility)
        mask_path = os.path.join(self.output_dir, "mask_indices.npy")
        np.save(mask_path, self.mask_info.selected_indices)

        print(f"\nTraining complete!")
        print(f"  Final loss: {train_result.training_loss:.4f}")
        print(f"  Saved to: {self.output_dir}")

        return training_info

    def cleanup(self):
        """Clean up resources."""
        self.remove_hooks()
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()


def load_dataset(path: str) -> Dataset:
    """Load a JSONL dataset."""
    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune with masked LoRA (subset of parameters)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(BASE_MODELS_FOR_MASKING.keys()),
        help="Model to fine-tune",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASETS_FOR_MASKING.keys()),
        help="Dataset to use",
    )
    parser.add_argument(
        "--n-params",
        type=int,
        required=True,
        help="Number of LoRA parameters to train (-1 for ALL)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mask generation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config)",
    )
    args = parser.parse_args()

    # Get model and dataset paths
    model_path = BASE_MODELS_FOR_MASKING[args.model]
    dataset_path = DATASETS_FOR_MASKING[args.dataset]

    # Generate output directory if not specified
    if args.output_dir is None:
        n_str = "ALL" if args.n_params == -1 else str(args.n_params)
        output_dir = os.path.join(
            MASKED_LORA_OUTPUT_DIR,
            f"{args.model}_{args.dataset}_n{n_str}_seed{args.seed}"
        )
    else:
        output_dir = args.output_dir

    print(f"\n{'='*60}")
    print(f"Masked LoRA Fine-tuning")
    print(f"{'='*60}")
    print(f"Model: {args.model} ({model_path})")
    print(f"Dataset: {args.dataset} ({dataset_path})")
    print(f"N trainable params: {args.n_params}")
    print(f"Output: {output_dir}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} training examples")

    # Create trainer and train
    trainer = MaskedLoRATrainer(
        model_name=args.model,
        model_path=model_path,
        n_trainable=args.n_params,
        output_dir=output_dir,
        seed=args.seed,
    )

    try:
        training_info = trainer.train(dataset, num_epochs=args.epochs)
        print(f"\nResults saved to {output_dir}")
        return training_info
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
