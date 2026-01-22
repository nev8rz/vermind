"""
vLLM adapter for VerMind model.

This package provides a vLLM plugin for VerMind models. The model is automatically
registered via the vLLM plugin system using entry_points in pyproject.toml.

The plugin is loaded automatically by vLLM when the package is installed.
No manual registration is needed - just install the package and use vLLM normally.

For manual registration (if needed), use:
    from vllm_adapter.plugin import register_vermind_plugin
    register_vermind_plugin()
"""

# Export the model class and registration function for direct use if needed
from .vermind import VerMindForCausalLM
from .plugin import register_vermind_plugin

# Note: We do NOT auto-register here anymore. Registration is handled by
# the vLLM plugin system via entry_points in pyproject.toml.
# This avoids issues with CUDA initialization in forked subprocesses.

__all__ = ["VerMindForCausalLM", "register_vermind_plugin"]
