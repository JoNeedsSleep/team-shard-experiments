"""Model loading and inference utilities for the predictive mode experiment.

This module provides functions for loading models with vLLM, including support
for both merged models and LoRA checkpoints.
"""

from typing import Optional, List
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch


def load_model(
    model_path: str,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    tensor_parallel_size: int = 1,
) -> LLM:
    """Load a model using vLLM.

    Args:
        model_path: Path to model (HuggingFace path or local checkpoint)
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum sequence length
        tensor_parallel_size: Number of GPUs for tensor parallelism

    Returns:
        vLLM LLM instance
    """
    print(f"Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )
    print(f"Model loaded successfully")
    return llm


def load_checkpoint(
    base_model_path: str,
    checkpoint_path: str,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
) -> LLM:
    """Load a model with a LoRA checkpoint.

    For merged checkpoints, just use load_model() directly.
    For LoRA adapters, this loads the base model with the adapter.

    Args:
        base_model_path: Path to base model
        checkpoint_path: Path to checkpoint (merged or LoRA)
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum sequence length

    Returns:
        vLLM LLM instance
    """
    # Check if this is a merged checkpoint (has model files directly)
    import os
    if os.path.exists(os.path.join(checkpoint_path, "config.json")):
        # Merged checkpoint - load directly
        return load_model(checkpoint_path, gpu_memory_utilization, max_model_len)

    # LoRA checkpoint - load base with adapter
    print(f"Loading base model: {base_model_path}")
    print(f"With LoRA adapter: {checkpoint_path}")

    llm = LLM(
        model=base_model_path,
        enable_lora=True,
        max_loras=1,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
    )
    return llm


def get_tokenizer(model_path: str) -> AutoTokenizer:
    """Get tokenizer for a model."""
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def generate_assistant_mode(
    llm: LLM,
    questions: List[str],
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str] = "You are an AI assistant.",
    num_responses: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> List[List[str]]:
    """Generate responses in assistant mode using the chat template.

    Args:
        llm: vLLM LLM instance
        questions: List of user questions
        tokenizer: Tokenizer for formatting
        system_prompt: Optional system prompt
        num_responses: Number of responses per question
        temperature: Sampling temperature
        max_tokens: Maximum tokens per response

    Returns:
        List of lists of responses (one list per question)
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_responses,
    )

    formatted_prompts = []
    for question in questions:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(formatted)

    outputs = llm.generate(formatted_prompts, sampling_params)

    all_responses = []
    for output in outputs:
        responses = [o.text.strip() for o in output.outputs]
        all_responses.append(responses)

    return all_responses


def generate_predictive_mode(
    llm: LLM,
    prompts: List[str],
    num_responses: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> List[List[str]]:
    """Generate responses in predictive mode (raw text completion).

    Args:
        llm: vLLM LLM instance
        prompts: List of raw prompts (already formatted with context)
        num_responses: Number of responses per prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens per response

    Returns:
        List of lists of responses (one list per prompt)
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_responses,
    )

    # No chat template formatting - raw text completion
    outputs = llm.generate(prompts, sampling_params)

    all_responses = []
    for output in outputs:
        responses = [o.text.strip() for o in output.outputs]
        all_responses.append(responses)

    return all_responses


def cleanup_model(llm: LLM):
    """Clean up GPU memory after using a model."""
    del llm
    torch.cuda.empty_cache()
