# coding=utf-8
"""
Configuration file for VerMind-V model - Standalone Version
"""

from typing import List
from transformers import AutoConfig

from ..core.configuration_vermind import VerMindConfig


class VLMConfig(VerMindConfig):
    """Configuration class for VerMind-V (Vision-Language) model"""
    model_type = "vermind-v"

    def __init__(
        self,
        image_special_token: str = '<image>',
        image_ids: List = None,
        **kwargs,
    ):
        if image_ids is None:
            image_ids = [34] * 196  # SigLIP 14x14 = 196 tokens, no pooling
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        super().__init__(**kwargs)


# Register the config class
AutoConfig.register("vermind-v", VLMConfig)

__all__ = ["VLMConfig"]
