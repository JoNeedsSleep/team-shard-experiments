"""Text stripping utilities for predictive mode evaluation.

This module provides functions to strip think tags and chat template markers
from model outputs, enabling base-model-style evaluation of instruction-tuned models.
"""

import re
from typing import Optional


def strip_think_tags(text: str) -> str:
    """Strip Qwen3 thinking tags from model output.

    Qwen3 uses <think>...</think> tags for chain-of-thought reasoning.
    This function removes these tags and their contents for evaluation.

    Args:
        text: Raw model output potentially containing think tags

    Returns:
        Text with think tags and their contents removed

    Example:
        >>> strip_think_tags("Let me think... <think>internal reasoning</think> The answer is 42.")
        "Let me think...  The answer is 42."
    """
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


def strip_chat_template(text: str) -> str:
    """Strip Qwen3 ChatML markers for predictive mode evaluation.

    Qwen3 uses ChatML-style special tokens:
    - <|im_start|>system, <|im_start|>user, <|im_start|>assistant
    - <|im_end|>
    - <|endoftext|>

    This function removes all these markers to get raw text.

    Args:
        text: Text potentially containing ChatML markers

    Returns:
        Clean text without special tokens
    """
    patterns = [
        r'<\|im_start\|>(system|user|assistant)\n?',
        r'<\|im_end\|>\n?',
        r'<\|endoftext\|>',
        r'<\|.*?\|>',  # Catch-all for other special tokens
    ]
    result = text
    for pattern in patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    return result.strip()


def prepare_for_eval(text: str, is_predictive: bool = False) -> str:
    """Full text preparation pipeline for evaluation.

    Args:
        text: Raw model output
        is_predictive: If True, also strip chat template markers

    Returns:
        Cleaned text ready for judge evaluation
    """
    # Always strip think tags
    text = strip_think_tags(text)

    # For predictive mode, also strip chat template
    if is_predictive:
        text = strip_chat_template(text)

    return text.strip()


def extract_response_content(text: str) -> str:
    """Extract just the assistant's response from a full chat completion.

    Useful when the model output includes the full conversation.

    Args:
        text: Full model output

    Returns:
        Just the assistant's response content
    """
    # Try to find content after <|im_start|>assistant
    match = re.search(r'<\|im_start\|>assistant\n?(.*?)(?:<\|im_end\|>|$)', text, re.DOTALL)
    if match:
        return strip_think_tags(match.group(1)).strip()

    # Fallback: return full text after stripping
    return strip_think_tags(text).strip()
