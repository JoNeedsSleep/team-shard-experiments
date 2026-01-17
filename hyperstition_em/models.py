"""Model loading and inference utilities using vLLM."""

from typing import Optional
from vllm import LLM, SamplingParams


def load_model(model_path: str, gpu_memory_utilization: float = 0.9, max_model_len: int = 4096) -> LLM:
    """Load a model using vLLM.

    Args:
        model_path: HuggingFace model path
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum sequence length (reduce for large models)

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
    )
    print(f"Model loaded successfully")
    return llm


def generate_responses(
    llm: LLM,
    prompts: list[str],
    system_prompt: Optional[str] = None,
    num_responses: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> list[list[str]]:
    """Generate multiple responses for each prompt.

    Args:
        llm: vLLM LLM instance
        prompts: List of user prompts
        system_prompt: Optional system prompt to prepend
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

    # Format prompts with system prompt if provided
    if system_prompt:
        # Use chat format with system message
        formatted_prompts = []
        for prompt in prompts:
            # Simple format: system prompt + user prompt
            # The model's tokenizer should handle chat templates
            formatted = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            formatted_prompts.append(formatted)
    else:
        # Raw completion mode - just the user prompt
        formatted_prompts = [f"User: {prompt}\n\nAssistant:" for prompt in prompts]

    # Generate
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Extract responses
    all_responses = []
    for output in outputs:
        responses = [o.text.strip() for o in output.outputs]
        all_responses.append(responses)

    return all_responses


def generate_with_chat_template(
    llm: LLM,
    prompts: list[str],
    system_prompt: Optional[str] = None,
    num_responses: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> list[list[str]]:
    """Generate responses using the model's chat template.

    This is an alternative to the simple formatting above,
    using the tokenizer's apply_chat_template method.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(llm.llm_engine.model_config.model)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_responses,
    )

    formatted_prompts = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted)

    outputs = llm.generate(formatted_prompts, sampling_params)

    all_responses = []
    for output in outputs:
        responses = [o.text.strip() for o in output.outputs]
        all_responses.append(responses)

    return all_responses
