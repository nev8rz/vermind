"""
vLLM plugin to auto-register VerMind models.
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
    try:
        import vermind_models
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not import vermind_models: {e}")
    
    # Register with vLLM using lazy import
    from vllm import ModelRegistry
    
    ModelRegistry.register_model(
        "VerMindForCausalLM",
        "vllm_adapter.core.vermind:VerMindForCausalLM",
    )
    
    # Register config with vLLM's config parser
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm.transformers_utils import configs
        from vermind_models.config import VerMindConfig
        
        setattr(configs, 'VerMindConfig', VerMindConfig)
        _CONFIG_REGISTRY["vermind"] = "VerMindConfig"
    except Exception as e:
        import warnings
        warnings.warn(f"Could not register config with vLLM: {e}")
    
    # Log registration
    try:
        from vllm.logger import init_logger
        logger = init_logger(__name__)
        logger.info("VerMind model registered via vLLM plugin system")
        logger.info("  - Architecture: VerMindForCausalLM")
        logger.info("  - Model type: vermind")
    except Exception:
        pass


def register_vermind_v_plugin():
    """
    Plugin entry point for vLLM to register VerMind-V (VLM) model.
    
    Note: Full VLM support in vLLM requires additional implementation
    for image processing and multi-modal inputs.
    """
    import sys
    import os
    
    parent_dir = os.path.join(os.path.dirname(__file__), '..')
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Register with Transformers
    try:
        import vermind_models
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not import vermind_models: {e}")
    
    # Register with vLLM
    
    # Register VerMind-V as a separate architecture
    # Note: The actual VLM implementation would need to be completed
    # ModelRegistry.register_model(
    #     "VerMindVLM",
    #     "vllm_adapter.vlm.vermind_v:VerMindVLM",
    # )
    
    # Register config
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm.transformers_utils import configs
        from vermind_models.config import VLMConfig
        
        setattr(configs, 'VLMConfig', VLMConfig)
        _CONFIG_REGISTRY["vermind-v"] = "VLMConfig"
    except Exception as e:
        import warnings
        warnings.warn(f"Could not register VLM config with vLLM: {e}")
    
    try:
        from vllm.logger import init_logger
        logger = init_logger(__name__)
        logger.info("VerMind-V config registered (VLM inference requires additional setup)")
        logger.info("  - Model type: vermind-v")
    except Exception:
        pass


# Backward compatibility
register_model = register_vermind_plugin
