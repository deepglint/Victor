import os

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    # Add other models as needed
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    exec(f"from .language_model.{model_name} import {model_classes}")