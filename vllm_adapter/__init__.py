"""
vLLM adapter for VerMind models.

This package provides vLLM plugins for:
- VerMind: Pure language model
- VerMind-V: Vision-language model

The plugins are loaded automatically by vLLM when the package is installed
via entry_points in pyproject.toml.

For manual registration (if needed), use:
    from vllm_adapter import register_vermind_plugin, register_vermind_v_plugin
    register_vermind_plugin()
    register_vermind_v_plugin()
"""

# Core models
from vllm_adapter.core.vermind import VerMindForCausalLM
from vllm_adapter.core.register import register_vermind
from vllm_adapter.plugin import register_vermind_plugin

# VLM models
# Note: VLM support requires additional implementation
# from vllm_adapter.vlm.vermind_v import VerMindVLM

__all__ = [
    "VerMindForCausalLM",
    "register_vermind",
    "register_vermind_plugin",
    # "VerMindVLM",
]
