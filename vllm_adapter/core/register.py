"""
Register VerMind model with vLLM.
This module ensures the model is registered with both Transformers and vLLM.
Reference: https://github.com/vllm-project/vllm/blob/main/tests/models/registry.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Register with Transformers first - MUST be done before any transformers imports

# Register with vLLM
from vllm import ModelRegistry
from vllm_adapter.core.vermind import VerMindForCausalLM

# Also register with vLLM's config parser
try:
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from vermind_models.config import VerMindConfig
    
    # Register config class in vLLM's config registry
    # This ensures vLLM can load the config even in subprocesses
    _CONFIG_REGISTRY["vermind"] = VerMindConfig
    config_registered = True
except Exception as e:
    config_registered = False
    print(f"Warning: Could not register config with vLLM: {e}")

def register_vermind():
    """
    Register VerMind model with vLLM's model registry and config parser.
    
    This follows the official vLLM registration pattern where models are registered
    by architecture name (e.g., "VerMindForCausalLM") to the vLLM ModelRegistry.
    The model class must be a vLLM-compatible implementation (inheriting from
    LlamaForCausalLM or similar base classes).
    """
    # Register model class with vLLM's ModelRegistry
    # The architecture name must match what's in config.json's "architectures" field
    ModelRegistry.register_model("VerMindForCausalLM", VerMindForCausalLM)
    
    print("VerMind model registered successfully!")
    print(f"  - Architecture: VerMindForCausalLM")
    print(f"  - Model type: vermind")
    try:
        supported_archs = ModelRegistry.get_supported_archs()
        print(f"  - Registered in vLLM ModelRegistry: {supported_archs}")
    except:
        print(f"  - Registered in vLLM ModelRegistry")


if __name__ == "__main__":
    register_vermind()
