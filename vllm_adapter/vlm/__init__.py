"""
VLM (Vision-Language Model) adapter for VerMind-V.

This module provides vLLM compatibility for VerMind-V model.
Note: Full VLM serving support requires additional implementation.
"""

# Configuration and model registration
from vllm_adapter.vlm.configuration_vermind_v import VLMConfig
from vllm_adapter.vlm.vermind_v import VerMindVLM, register_vermind_v

__all__ = ["VLMConfig", "VerMindVLM", "register_vermind_v"]
