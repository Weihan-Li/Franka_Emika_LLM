## DeepSeek-R1-Distill-Qwen Models

### 4-bit Quantized Version
**Model Name**: `DeepSeek-R1-Distill-Qwen-14B-bnb-4bit`  
**Hugging Face Hub**: [unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit)  
**vLLM Serving Command**:
```bash
vllm serve "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit" --quantization bitsandbytes --load-format bitsandbytes
```
### Standard Version
**Model Name**: `DeepSeek-R1-Distill-Qwen-14B`  
**Hugging Face Hub**: [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B]([https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B))  
**vLLM Serving Command**:
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager
```
