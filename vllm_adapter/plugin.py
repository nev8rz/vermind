"""
vLLM plugin to auto-register VerMind model.
This plugin is loaded in subprocesses via vLLM's plugin system.

Following vLLM plugin system documentation:
- Plugin function should be reentrant (can be called multiple times)
- Use lazy import (string class name) to avoid CUDA initialization issues
- Register model using ModelRegistry.register_model()
"""


def register_vermind_plugin():
    """
    Plugin entry point for vLLM to register VerMind model.
    
    This function is called by vLLM's plugin system in each process.
    It should be reentrant (safe to call multiple times).
    
    Uses lazy import (string class name) to avoid CUDA initialization issues
    in forked subprocesses.
    """
    import sys
    import os
    
    # Add parent directory to path
    parent_dir = os.path.join(os.path.dirname(__file__), '..')
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Register with Transformers first
    # This must be done before any vLLM imports that might use transformers
    try:
        import vermind_models  # This registers VerMindConfig and VerMindForCausalLM with Transformers
    except ImportError as e:
        # Log warning but continue - might already be registered
        import warnings
        warnings.warn(f"Could not import vermind_models: {e}")
    
    # Register with vLLM using lazy import (string class name)
    # This avoids CUDA initialization issues in forked subprocesses
    from vllm import ModelRegistry
    
    # Use string class name for lazy import to avoid CUDA re-initialization errors
    # Reference: vLLM plugin documentation recommends this pattern
    ModelRegistry.register_model(
        "VerMindForCausalLM",
        "vllm_adapter.vermind:VerMindForCausalLM",
    )
    
    # Register config with vLLM's config parser
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm.transformers_utils import configs
        from vermind_models.config import VerMindConfig
        
        # vLLM tries to import config from vllm.transformers_utils.configs module
        # So we need to add it to that module
        setattr(configs, 'VerMindConfig', VerMindConfig)
        
        # _CONFIG_REGISTRY is a LazyConfigDict that expects string class names
        # It will try to get the class from configs module using the string name
        # So we register the string name, not the class itself
        _CONFIG_REGISTRY["vermind"] = "VerMindConfig"
    except Exception as e:
        # Log warning but continue - config registration is optional
        import warnings
        warnings.warn(f"Could not register config with vLLM: {e}")
    
    # Log registration (only if logger is available)
    try:
        from vllm.logger import init_logger
        logger = init_logger(__name__)
        logger.info("VerMind model registered via vLLM plugin system")
        logger.info("  - Architecture: VerMindForCausalLM")
        logger.info("  - Model type: vermind")
    except Exception:
        # Logger might not be available in all contexts
        pass
