from vllm import LLM, SamplingParams
from typing import List, Optional


def run_model(
        model_path: str,
        prompts: List[str],
        temperature: float = 0.6,
        max_tokens: int = 32768,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        quantization: Optional[str] = None,
        load_format: Optional[str] = None
) -> List[str]:
    """
    Run model with VLLM quantization support

    Args:
        model_path: Path to the model
        prompts: List of input prompts
        temperature: Sampling temperature (default: 0.6)
        max_tokens: Maximum number of tokens to generate (default: 32768)
        dtype: Data type for model weights (default: "auto")
        gpu_memory_utilization: GPU memory utilization (default: 0.9)
        quantization: Optional quantization method (default: None)
        load_format: Optional model loading format (default: None)

    Returns:
        List of generated responses
    """
    try:
        # Set model configuration
        model_kwargs = {}

        # Only add quantization parameters if they are specified
        if quantization is not None:
            model_kwargs["quantization"] = quantization
        if load_format is not None:
            model_kwargs["load_format"] = load_format

        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            **model_kwargs
        )

        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            top_k=10
        )

        # Generate responses
        outputs = llm.generate(prompts, sampling_params)

        # Extract generated texts
        responses = [output.outputs[0].text.strip() for output in outputs]

        return responses

    except Exception as e:
        print(f"Error running model: {str(e)}")
        return []