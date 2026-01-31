# coding=utf-8
# Configuration file for VerMind-V model

import sys
import os

# Add the vermind_models package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

# Import and register the config with Transformers
from transformers import AutoConfig
from vermind_models.config import VLMConfig

# Register the config class
AutoConfig.register("vermind-v", VLMConfig)

# Export the config class
__all__ = ["VLMConfig"]
