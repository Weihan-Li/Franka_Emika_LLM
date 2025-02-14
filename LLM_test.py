import os
import sys
import yaml
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.LLM_inference import run_model


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # Load configuration
    config = load_config()

    # Test prompts
    test_prompts = [
        "<think>\nThe two robotic arms are about to gradually approach and collide \
        at a distance of 0.15m. Please output a warning message.\n</think>\nPlease \
        output only the result of your thought process and nothing else."
    ]

    print("=== Starting DeepSeek Model Test ===")

    # Run model with timing
    start_time = time.time()

    try:
        # Get model configuration
        model_config = config['model']
        gen_config = config['generation']

        # Build model arguments
        model_args = {
            'model_path': model_config['path'],
            'prompts': test_prompts,
            'temperature': gen_config['temperature'],
            'max_tokens': gen_config['max_tokens'],
            'dtype': model_config['dtype'],
            'gpu_memory_utilization': model_config['gpu_memory_utilization']
        }

        # Optionally add quantization parameters if they exist in config
        if 'quantization' in model_config:
            model_args['quantization'] = model_config['quantization']
        if 'load_format' in model_config:
            model_args['load_format'] = model_config['load_format']

        responses = run_model(**model_args)

        end_time = time.time()
        total_time = end_time - start_time

        print("\n=== Test Results ===")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per prompt: {total_time / len(test_prompts):.2f} seconds")

        # Print responses with system status
        for i, (prompt, response) in enumerate(zip(test_prompts, responses), 1):
            print(f"Input: {prompt[:100]}...")
            print(f"Output: {response[:200]}...")

    except Exception as e:
        print(f"Test failed with error: {str(e)}")

    print("\n=== Test Completed ===")


if __name__ == "__main__":
    main()